; Установщик Arx Nova для Windows (Inno Setup). Собирается на GitHub: iscc windows\arxnova.iss
; Ставится в профиль пользователя — права администратора не нужны.
#define AppVer GetEnv("ARXNOVA_VERSION")
[Setup]
AppId={{D812ED8E-076A-42DB-ADA4-FAD9892F8136}
AppName=Arx Nova
AppVersion={#AppVer}
AppPublisher=Arx
DefaultDirName={localappdata}\Programs\Arx Nova
DefaultGroupName=Arx Nova
PrivilegesRequired=lowest
OutputDir=..\dist
OutputBaseFilename=ArxNova-{#AppVer}-Windows-x64-setup
SetupIconFile=..\icon.ico
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
DisableProgramGroupPage=yes
ArchitecturesInstallIn64BitMode=x64compatible

[Languages]
Name: "ru"; MessagesFile: "compiler:Languages\Russian.isl"
Name: "en"; MessagesFile: "compiler:Default.isl"

[Files]
Source: "..\dist\Arx Nova\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion

[Icons]
Name: "{userprograms}\Arx Nova"; Filename: "{app}\Arx Nova.exe"
Name: "{userdesktop}\Arx Nova"; Filename: "{app}\Arx Nova.exe"

[Run]
Filename: "{app}\Arx Nova.exe"; Description: "{cm:LaunchProgram,Arx Nova}"; Flags: nowait postinstall skipifsilent

[UninstallRun]
Filename: "{app}\Arx Nova.exe"; Parameters: "service remove"; Flags: runhidden; RunOnceId: "RemoveService"
