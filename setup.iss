#define MyAppName "NinuGaming Live Assistant"
#define MyAppVersion "2.2.0-rc.2"
#define MyAppPublisher "NinuGaming"
#define MyAppExeName "NinuGamingLiveAssistant.exe"

[Setup]
AppId={{7A2D7A7A-AD17-4D46-9B5C-0CCB4774A56F}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL=https://github.com/NinuGaming999/Yt-chat-reader
DefaultDirName={autopf}\NinuGaming Live Assistant
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64
PrivilegesRequired=admin
OutputDir=installer
OutputBaseFilename=NinuGamingLiveAssistant-Setup
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
SetupIconFile=app.ico
UninstallDisplayIcon={app}\{#MyAppExeName}
CloseApplications=yes
RestartApplications=no
SetupLogging=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional shortcuts:"; Flags: unchecked

[Dirs]
Name: "{commonappdata}\NinuGamingLiveAssistant"; Permissions: users-modify
Name: "{commonappdata}\NinuGamingLiveAssistant\models"; Permissions: users-modify
Name: "{commonappdata}\NinuGamingLiveAssistant\runtime"; Permissions: users-modify

[Files]
Source: "dist\NinuGamingLiveAssistant\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "installer_assets\vc_redist.x64.exe"; DestDir: "{tmp}"; Flags: deleteafterinstall
Source: "installer_assets\MicrosoftEdgeWebView2Setup.exe"; DestDir: "{tmp}"; Flags: deleteafterinstall
Source: "README.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "THIRD_PARTY_NOTICES.md"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\NinuGaming Live Assistant"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\NinuGaming Live Assistant"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{tmp}\vc_redist.x64.exe"; Parameters: "/install /quiet /norestart"; StatusMsg: "Installing Microsoft Visual C++ runtime..."; Flags: waituntilterminated; Check: not VCRedistInstalled
Filename: "{tmp}\MicrosoftEdgeWebView2Setup.exe"; Parameters: "/silent /install"; StatusMsg: "Installing Microsoft Edge WebView2 runtime..."; Flags: waituntilterminated; Check: not WebView2AlreadyInstalled
Filename: "{app}\{#MyAppExeName}"; Description: "Launch NinuGaming Live Assistant"; Flags: postinstall nowait skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}"

[Code]
function VCRedistInstalled: Boolean;
var
  Installed: Cardinal;
begin
  Result := RegQueryDWordValue(HKLM64, 'SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64', 'Installed', Installed) and (Installed = 1);
end;

function WebView2AlreadyInstalled: Boolean;
begin
  Result := DirExists(ExpandConstant('{pf64}\Microsoft\EdgeWebView\Application')) or
            DirExists(ExpandConstant('{pf32}\Microsoft\EdgeWebView\Application'));
end;