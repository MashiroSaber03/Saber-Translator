//! XInput2 raw tablet events on X11, over a separate pure-Rust X connection (x11rb's
//! `RustConnection`, no `unsafe`, no libxcb).
//!
//! winit selects XI2 pointer events on its window for the master pointer, whose events carry the
//! valuators of whichever slave moved but whose valuator layout changes with it, and which go to
//! winit alone while a drag holds the implicit grab. Raw events (`XI_RawMotion`,
//! `XI_RawButtonPress/Release`) selected on the root window have neither problem: since XI 2.1
//! every client that selects them gets them, grabbed or not, tagged with the slave device. So
//! this thread listens to those, plus hierarchy and device changes to keep the device list
//! current, and maps them with [`crate::xi`].
//!
//! Raw events don't say which window the pen is over. That's fine for pressure: the UI only uses
//! the sample while its own canvas receives pointer events, and any other pointer moving resets
//! it to "mouse".

use std::thread::JoinHandle;

use x11rb::connection::Connection;
use x11rb::protocol::Event;
use x11rb::protocol::xinput::{self, ConnectionExt as _, DeviceClassData, DeviceType, XIEventMask};
use x11rb::protocol::xproto::ConnectionExt as _;
use x11rb::rust_connection::RustConnection;

use crate::xi::{Device, State, fp3232, valuators};
use crate::{Error, Sample, Update, deliver};

/// `XIAllDevices`: select for every device, slaves included.
const ALL_DEVICES: u16 = 0;

/// Start listening on `display` (`None` = `$DISPLAY`, the display winit uses). The connection
/// and the XI2 version are checked before this returns; events are then read on a background
/// thread that calls `callback` with `Some(sample)` for pen events and `None` when another
/// pointer moves. The thread ends when the X connection closes.
pub fn spawn(display: Option<&str>, callback: impl Fn(Option<Sample>) + Send + 'static) -> Result<JoinHandle<()>, Error> {
    let (conn, screen) = RustConnection::connect(display).map_err(|e| Error::Unsupported(format!("cannot connect to the X server: {e}")))?;
    let root = conn.setup().roots.get(screen).map(|s| s.root).ok_or_else(|| Error::Platform(format!("X screen {screen} not found")))?;
    // Raw events for everyone (not just the grabbing client) need XI 2.1; ask for 2.2.
    let version =
        conn.xinput_xi_query_version(2, 2).map_err(|e| Error::Platform(e.to_string()))?.reply().map_err(|e| Error::Unsupported(format!("no XInput2: {e}")))?;
    if (version.major_version, version.minor_version) < (2, 1) {
        return Err(Error::Unsupported(format!("XInput {}.{} (raw events need 2.1)", version.major_version, version.minor_version)));
    }
    let mask = XIEventMask::RAW_MOTION | XIEventMask::RAW_BUTTON_PRESS | XIEventMask::RAW_BUTTON_RELEASE | XIEventMask::HIERARCHY | XIEventMask::DEVICE_CHANGED;
    conn.xinput_xi_select_events(root, &[xinput::EventMask { deviceid: ALL_DEVICES, mask: vec![mask] }])
        .map_err(|e| Error::Platform(e.to_string()))?
        .check()
        .map_err(|e| Error::Platform(format!("XISelectEvents: {e}")))?;
    let mut state = State::default();
    state.set_devices(query_devices(&conn));
    std::thread::Builder::new()
        .name("photocraft-tablet-x11".into())
        .spawn(move || run(&conn, &mut state, &callback))
        .map_err(|e| Error::Platform(format!("cannot start the tablet thread: {e}")))
}

fn run(conn: &RustConnection, state: &mut State, callback: &dyn Fn(Option<Sample>)) {
    // A connection error (the display went away) ends the loop; malformed events are skipped by
    // x11rb's parser and by the bounds in `xi`.
    while let Ok(event) = conn.wait_for_event() {
        let update = match event {
            Event::XinputRawMotion(e) | Event::XinputRawButtonPress(e) | Event::XinputRawButtonRelease(e) => {
                let values: Vec<f64> = e.axisvalues.iter().map(|v| fp3232(v.integral, v.frac)).collect();
                state.handle(e.deviceid, valuators(&e.valuator_mask, &values))
            }
            Event::XinputHierarchy(_) | Event::XinputDeviceChanged(_) => {
                state.set_devices(query_devices(conn));
                Update::Keep
            }
            _ => Update::Keep,
        };
        if let Update::Set(sample) = update {
            deliver(callback, sample);
        }
    }
}

/// The slave pointer devices with their pen valuators (empty on any error).
fn query_devices(conn: &RustConnection) -> Vec<Device> {
    let Ok(reply) = conn.xinput_xi_query_device(ALL_DEVICES).map_err(drop).and_then(|c| c.reply().map_err(drop)) else { return Vec::new() };
    let mut out = Vec::new();
    for info in reply.infos.iter().take(crate::xi::MAX_DEVICES) {
        if info.type_ != DeviceType::SLAVE_POINTER && info.type_ != DeviceType::FLOATING_SLAVE {
            continue;
        }
        let name = String::from_utf8_lossy(&info.name);
        let valuators = info.classes.iter().take(crate::xi::MAX_AXES).filter_map(|c| match &c.data {
            DeviceClassData::Valuator(v) => {
                let label = atom_name(conn, v.label)?;
                Some((v.number, label, fp3232(v.min.integral, v.min.frac), fp3232(v.max.integral, v.max.frac)))
            }
            _ => None,
        });
        out.push(Device::new(info.deviceid, &name, valuators.collect::<Vec<_>>()));
    }
    out
}

fn atom_name(conn: &RustConnection, atom: u32) -> Option<String> {
    if atom == 0 {
        return None;
    }
    let reply = conn.get_atom_name(atom).ok()?.reply().ok()?;
    Some(String::from_utf8_lossy(&reply.name).into_owned())
}
