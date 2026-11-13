# Build a Windows installer

Everything here targets the packaged folder, not the repository:

```
dist\PrivateVoice_Ahee52\        prepared application folder (app, assets, runtime, models)
packaging\installer.iss          Inno Setup script (paths are absolute for this machine)
```

## Requirements

- Inno Setup 6 (`ISCC.exe`)
- The prepared folder at `dist\PrivateVoice_Ahee52`

## Build

From the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_installer.ps1
```

Output goes to `dist\installer\` as one `.exe` plus 7 `.bin` slices (each < 2 GB so they can be
uploaded to GitHub Releases individually).

## Notes

- `installer.iss` uses absolute paths (`C:\Local Repositories\PrivateVoice\dist\...`). If you move
  the project, update **OutputDir**, **SetupIconFile**, **WizardImageFile** and the `[Files]`
  `Source:` line.
- The installer bundles the whisper model, so expect ~11.5 GB of output.
- Close any running `pythonw.exe` from the install folder first — Inno cannot overwrite locked files.

## Making an incremental update package

```powershell
powershell -ExecutionPolicy Bypass -File scripts\make_update_zip.ps1 -Version 1.0.5
```

Result: `dist\PrivateVoice_Ahee52-<version>-update.zip` containing only
`app/ assets/ tools/ runtime/ README.txt app.ico` — no models, no character data.
