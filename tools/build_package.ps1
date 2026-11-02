# 情感陪伴器_v1 打包脚本（Stage 4）
# 用法：pwsh -File tools\build_package.ps1
#
# 做的事：
#   1. 下载 python-3.12.10-embed-amd64.zip 到 dist\
#   2. 建 dist\情感陪伴器_v1\，解压运行时，写 python312._pth（含 import site）
#   3. 拷 site-packages、app、assets、Aqua-TTS、GPT-SoVITS、characters
#   4. 删掉 editable 安装残留（写死绝对路径）
#   5. 生成启动/自检 bat 与图标，调用 ISCC 编译安装器（分卷）
#
# 注意：产物约 9.5GB，dist\ 已在 .gitignore 里。

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Dist = Join-Path $Root "dist"
$Pkg = Join-Path $Dist "情感陪伴器_v1"
$Py = Join-Path $Root ".venv\Scripts\python.exe"
$EmbedZip = Join-Path $Dist "python-embed.zip"
$EmbedUrl = "https://www.python.org/ftp/python/3.12.10/python-3.12.10-embed-amd64.zip"

Write-Host "== 1/6 准备目录 =="
New-Item -ItemType Directory -Force -Path $Dist, $Pkg | Out-Null

if (-not (Test-Path $EmbedZip)) {
    Write-Host "  下载嵌入式 Python"
    Invoke-WebRequest -Uri $EmbedUrl -OutFile $EmbedZip
}
if (-not (Test-Path (Join-Path $Pkg "runtime\python.exe"))) {
    Expand-Archive -Path $EmbedZip -DestinationPath (Join-Path $Pkg "runtime") -Force
}

Write-Host "== 2/6 写 ._pth（必须有 import site）=="
$pth = @(
    "python312.zip",
    ".",
    "Lib\site-packages",
    "..",
    "..\Aqua-TTS",
    "import site"
) -join "`r`n"
Set-Content -Path (Join-Path $Pkg "runtime\python312._pth") -Value $pth -Encoding ascii

Write-Host "== 3/6 拷依赖与代码 =="
robocopy (Join-Path $Root ".venv\Lib\site-packages") (Join-Path $Pkg "runtime\Lib\site-packages") /E /MT:16 /NFL /NDL /NJH /NJS /R:1 /W:1 | Out-Null
robocopy (Join-Path $Root "app") (Join-Path $Pkg "app") /E /NFL /NDL /NJH /NJS /XD __pycache__ | Out-Null
robocopy (Join-Path $Root "assets") (Join-Path $Pkg "assets") /E /NFL /NDL /NJH /NJS | Out-Null
robocopy (Join-Path $Root "tools") (Join-Path $Pkg "tools") selftest.py /NFL /NDL /NJH /NJS | Out-Null
robocopy (Join-Path $Root "Aqua-TTS") (Join-Path $Pkg "Aqua-TTS") /E /NFL /NDL /NJH /NJS /XD .git __pycache__ | Out-Null
robocopy (Join-Path $Root "GPT-SoVITS") (Join-Path $Pkg "GPT-SoVITS") /E /MT:16 /NFL /NDL /NJH /NJS /R:1 /W:1 /XD .git __pycache__ /XF G2PWModel.zip | Out-Null
robocopy (Join-Path $Root "characters") (Join-Path $Pkg "characters") /E /NFL /NDL /NJH /NJS /XD __pycache__ | Out-Null
foreach ($d in "logs", "datasets", "results") {
    New-Item -ItemType Directory -Force -Path (Join-Path $Pkg $d) | Out-Null
}

Write-Host "== 4/6 清掉 editable 安装残留 =="
Get-ChildItem (Join-Path $Pkg "runtime\Lib\site-packages") -Filter "__editable__*" -ErrorAction SilentlyContinue |
    Remove-Item -Force

Write-Host "== 5/6 角色路径相对化（换机回落）=="
& $Py -c @"
import json, pathlib
base = pathlib.Path(r'$Pkg\characters')
for p in base.glob('*/profile.json'):
    d = json.loads(p.read_text(encoding='utf-8'))
    v = d.get('voice', {})
    for k in ('gpt_ckpt', 'sovits_ckpt'):
        if v.get(k):
            v[k] = 'checkpoints/' + pathlib.Path(v[k]).name
    d['voice'] = v
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding='utf-8')
print('  characters relocated')
"@

Write-Host "== 6/6 编译安装器 =="
$iscc = Join-Path $env:LOCALAPPDATA "Programs\Inno Setup 6\ISCC.exe"
if (Test-Path $iscc) {
    & $iscc /Qp (Join-Path $Root "installer.iss")
    Write-Host "  安装器输出：$Dist\installer"
} else {
    Write-Host "  未找到 ISCC.exe，跳过安装器（便携版已就绪：$Pkg）"
}

Write-Host "完成。自检：cd `"$Pkg`"; runtime\python.exe tools\selftest.py"
