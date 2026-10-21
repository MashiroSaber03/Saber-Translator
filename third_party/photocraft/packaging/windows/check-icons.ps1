<#
.SYNOPSIS
  Check MSI icon references without building or installing PhotoCraft.

.DESCRIPTION
  ICE50 requires advertised shortcut icon identifiers to have the target file's extension.
  Windows Installer uses those identifiers as cached filenames; extensionless identifiers
  can display a generic document icon even when the ICO data and application are valid.
#>
param(
  [string] $WxsPath = (Join-Path $PSScriptRoot 'photocraft.wxs')
)
$ErrorActionPreference = 'Stop'
[xml] $Wxs = Get-Content -LiteralPath $WxsPath -Raw
$Ns = [Xml.XmlNamespaceManager]::new($Wxs.NameTable)
$Ns.AddNamespace('w', 'http://wixtoolset.org/schemas/v4/wxs')
$Icons = @{}
foreach ($icon in $Wxs.SelectNodes('//w:Icon', $Ns)) {
  $Icons[$icon.GetAttribute('Id')] = $icon
}

foreach ($shortcut in $Wxs.SelectNodes('//w:File/w:Shortcut[@Advertise="yes"]', $Ns)) {
  $id = $shortcut.GetAttribute('Id')
  $iconId = $shortcut.GetAttribute('Icon')
  if (-not $Icons.ContainsKey($iconId)) {
    throw "Advertised shortcut '$id' references missing icon '$iconId'."
  }
  $iconExtension = [IO.Path]::GetExtension($iconId)
  if ($iconExtension -notin '.exe', '.ico') {
    throw "Advertised shortcut '$id' icon '$iconId' must have an .exe or .ico extension (ICE50)."
  }
  $file = $shortcut.ParentNode
  $fileName = if ($file.HasAttribute('Name')) { $file.GetAttribute('Name') } else { $file.GetAttribute('Source') }
  # Source paths use Windows separators even when this check runs on Linux in packaging-lint.
  $fileName = ($fileName -split '[\\/]')[-1]
  $targetExtension = [IO.Path]::GetExtension($fileName)
  if ($iconExtension -ne $targetExtension) {
    throw "Advertised shortcut '$id' icon '$iconId' must match target '$fileName' extension '$targetExtension' (ICE50)."
  }
}

$arpIcon = $Wxs.SelectSingleNode('//w:Property[@Id="ARPPRODUCTICON"]', $Ns)
if ($arpIcon -and -not $Icons.ContainsKey($arpIcon.GetAttribute('Value'))) {
  throw "ARPPRODUCTICON references missing icon '$($arpIcon.GetAttribute('Value'))'."
}
Write-Output 'Windows shortcut icon references ok'
