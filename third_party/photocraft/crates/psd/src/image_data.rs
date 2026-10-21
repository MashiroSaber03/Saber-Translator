//! Merged (composite) image data section.

use crate::compression::{Compression, PlaneLayout, decode_planes, encode_planes, validate_planes};
use crate::error::Result;
use crate::header::Header;
use crate::io::Reader;

/// The merged image: compression plus the encoded bytes of all channels
/// (to the end of the file), kept verbatim.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ImageData {
    /// Compression method for the whole image.
    pub compression: Compression,
    /// Encoded data. For RLE: the byte-count table for every row of every
    /// channel followed by the rows.
    pub data: Vec<u8>,
}

impl ImageData {
    /// Plane layout for the merged image described by `header`.
    pub fn layout(header: &Header) -> PlaneLayout {
        PlaneLayout {
            planes: usize::from(header.channels),
            width: header.width as usize,
            height: header.height as usize,
            depth: header.depth,
            version: header.version,
        }
    }

    /// Encodes planar data (`channels × height × row_bytes`).
    pub fn encode(compression: Compression, decoded: &[u8], header: &Header) -> Result<Self> {
        Ok(ImageData { compression, data: encode_planes(compression, decoded, &Self::layout(header))? })
    }

    /// Decodes all channels into planar big-endian samples.
    pub fn decode(&self, header: &Header) -> Result<Vec<u8>> {
        decode_planes(self.compression, &self.data, &Self::layout(header))
    }

    /// Decodes and returns a single channel plane.
    pub fn decode_channel(&self, header: &Header, index: usize) -> Result<Vec<u8>> {
        let invalid = || crate::error::PsdError::invalid(format!("no merged channel {index}"));
        if index >= usize::from(header.channels) {
            return Err(invalid());
        }
        let plane = header.row_bytes().checked_mul(header.height as usize).ok_or_else(invalid)?;
        let start = index.checked_mul(plane).ok_or_else(invalid)?;
        let end = start.checked_add(plane).ok_or_else(invalid)?;
        let all = self.decode(header)?;
        all.get(start..end).map(<[u8]>::to_vec).ok_or_else(invalid)
    }

    pub(crate) fn read(r: &mut Reader<'_>, header: &Header) -> Result<Self> {
        let compression = Compression::from_u16(r.u16()?);
        let data = r.peek_rest();
        validate_planes(compression, data, &Self::layout(header))?;
        let data = data.to_vec();
        r.skip(data.len())?;
        Ok(ImageData { compression, data })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::header::{ColorMode, Version};

    #[test]
    fn merged_channels_reject_invalid_indices_without_decoding_or_overflow() {
        let header = Header::new(Version::Psd, 2, 1, 3, 8, ColorMode::Rgb);
        let data = ImageData { compression: Compression::Raw, data: vec![10, 11, 20, 21, 30, 31] };
        for (index, expected) in [(0, [10, 11]), (1, [20, 21]), (2, [30, 31])] {
            assert_eq!(data.decode_channel(&header, index).unwrap(), expected.to_vec());
        }
        for index in [3, 4, usize::MAX] {
            assert!(data.decode_channel(&header, index).is_err(), "invalid channel {index}");
        }
        // Reject an invalid index before trying to decode corrupt merged data.
        let corrupt = ImageData { compression: Compression::Raw, data: vec![] };
        let err = corrupt.decode_channel(&header, usize::MAX).unwrap_err().to_string();
        assert!(err.contains("no merged channel"), "{err}");
    }
}
