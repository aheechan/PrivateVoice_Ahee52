; Ahee52_PrivateVoice-ASR-cn-beta 安装脚本（Inno Setup 6）——内置 GPT-SoVITS v4
; 编译：ISCC.exe /Qp installer.iss
; 说明：包体约 12.6GB，为了构建速度这里不压缩；安装时直接复制。

#define MyAppName "Ahee52_PrivateVoice-ASR-cn-beta"
#define MyAppVersion "2.0.0"
#define MyAppPublisher "VoiceLab"
#define MyAppExeName "Ahee52_PrivateVoice-ASR-cn-beta.exe"

[Setup]
AppId={{B7E4A1C9-52D3-4F18-9E06-3C8D1A7B5F42}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\Ahee52_PrivateVoice
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
OutputDir=C:\VoiceLab\dist\installer
OutputBaseFilename=Ahee52_PrivateVoice_Setup
SetupIconFile=C:\VoiceLab\dist\app.ico
UninstallDisplayIcon={app}\app.ico
Compression=none
SolidCompression=no
; Windows 单个 exe 上限约 4.2GB，包体 9.5GB 必须分卷
DiskSpanning=yes
DiskSliceSize=4000000000
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
PrivilegesRequired=admin
DisableDirPage=no

[Languages]
Name: "chinese"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "创建桌面快捷方式"; GroupDescription: "快捷方式："

[Files]
Source: "C:\VoiceLab\dist\Ahee52_PrivateVoice\*"; DestDir: "{app}"; \
  Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\runtime\pythonw.exe"; \
  Parameters: """{app}\app\main.py"""; WorkingDir: "{app}"; IconFilename: "{app}\app.ico"
Name: "{group}\自检"; Filename: "{app}\runtime\python.exe"; \
  Parameters: """{app}\tools\selftest.py"""; WorkingDir: "{app}"
Name: "{group}\使用说明"; Filename: "{app}\使用说明.txt"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\runtime\pythonw.exe"; \
  Parameters: """{app}\app\main.py"""; WorkingDir: "{app}"; IconFilename: "{app}\app.ico"; \
  Tasks: desktopicon

[Run]
Filename: "{app}\runtime\python.exe"; Parameters: """{app}\tools\selftest.py"""; \
  Description: "安装完成后立即自检（约 40 秒）"; Flags: postinstall nowait skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}\logs"
Type: filesandordirs; Name: "{app}\datasets"
