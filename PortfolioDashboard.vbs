Set WshShell = CreateObject("WScript.Shell")
Set objFSO = CreateObject("Scripting.FileSystemObject")

' Get the directory where this VBS file is located
strScriptPath = objFSO.GetParentFolderName(WScript.ScriptFullName)

' Change to the project directory
WshShell.CurrentDirectory = strScriptPath

' Set environment variables for unicode support
WshShell.Environment("Process")("PYTHONIOENCODING") = "utf-8"
WshShell.Environment("Process")("PYTHONUTF8") = "1"
WshShell.Environment("Process")("PYTHONLEGACYWINDOWSSTDIO") = "0"

' Run the Python script
WshShell.Run "python main.py", 1, False
