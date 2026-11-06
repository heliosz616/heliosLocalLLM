$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$Source = "$Root\vendor\llama.cpp"
$Commit = 'd81235049384534c167caea52b85a694f6103d14'
$VS = $env:SWIFT_VS_ROOT
if (-not $VS) { $VS = 'C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools' }
$Dev = "$VS\Common7\Tools\VsDevCmd.bat"
$CMake = "$VS\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
$Ninja = "$VS\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"
$Sdk = $env:VULKAN_SDK
$CudaArch = $env:SWIFT_CUDA_ARCH
if (-not $CudaArch) { $CudaArch = '120' }
if ($CudaArch -notmatch '^\d+$') { throw 'SWIFT_CUDA_ARCH must be numeric' }
foreach ($File in @($Dev,$CMake,$Ninja,"$Sdk\Bin\glslc.exe")) {
    if (-not (Test-Path -LiteralPath $File)) { throw "Missing build prerequisite: $File" }
}
if (-not (Test-Path "$Source\.git")) {
    git clone --depth 1 --branch b11429 https://github.com/ggml-org/llama.cpp.git $Source
    if ($LASTEXITCODE -ne 0) { throw 'Source checkout failed' }
}
if ((git -C $Source rev-parse HEAD) -ne $Commit) { throw 'Unexpected source revision' }
git -C $Source apply --reverse --check "$Root\patches\runtime.patch" 2>$null
if ($LASTEXITCODE -ne 0) {
    git -C $Source apply --check "$Root\patches\runtime.patch"
    if ($LASTEXITCODE -ne 0) { throw 'Source has conflicting changes; use a clean pinned checkout' }
    git -C $Source apply "$Root\patches\runtime.patch"
    if ($LASTEXITCODE -ne 0) { throw 'Runtime patch failed' }
}
$Common = '-DGGML_BACKEND_DL=ON -DGGML_NATIVE=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_CURL=OFF'
$Builds = @(
    @{Name='server'; Flags='-DGGML_CPU_ALL_VARIANTS=ON -DLLAMA_BUILD_TOOLS=ON'; Target='llama-server'; Jobs=6},
    @{Name='cuda'; Flags="-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=$CudaArch -DGGML_CUDA_FA_QUANTS=q5_1-q5_1,q4_0-q4_0,q8_0-q8_0,f16-f16 -DLLAMA_BUILD_TOOLS=OFF"; Target='ggml-cuda'; Jobs=2},
    @{Name='vulkan'; Flags=('-DGGML_VULKAN=ON -DCMAKE_PREFIX_PATH="{0}" -DLLAMA_BUILD_TOOLS=OFF' -f $Sdk); Target='ggml-vulkan'; Jobs=2}
)
foreach ($Build in $Builds) {
    $Dir = "$Root\build\$($Build.Name)"
    $Command = '"{0}" -arch=amd64 && "{1}" -S "{2}" -B "{3}" -G Ninja -DCMAKE_MAKE_PROGRAM="{4}" -DCMAKE_BUILD_TYPE=Release {5} {6} && "{1}" --build "{3}" --target {7} -j {8}' -f $Dev,$CMake,$Source,$Dir,$Ninja,$Common,$Build.Flags,$Build.Target,$Build.Jobs
    cmd.exe /d /s /c $Command
    if ($LASTEXITCODE -ne 0) { throw "Build failed: $($Build.Name)" }
}
$Runtime = "$Root\build\runtime"
New-Item -ItemType Directory -Force $Runtime | Out-Null
Copy-Item "$Root\build\server\bin\*.dll" $Runtime -Force
Copy-Item "$Root\build\server\bin\llama-server.exe" $Runtime -Force
Copy-Item "$Root\build\cuda\bin\ggml-cuda.dll" $Runtime -Force
Copy-Item "$Root\build\vulkan\bin\ggml-vulkan.dll" $Runtime -Force
$Target = "$Root\build\runtime-mtp"
New-Item -ItemType Directory -Force $Target | Out-Null
Copy-Item "$Runtime\*" $Target -Force
$Files = @(Get-ChildItem $Runtime -File | Where-Object { $_.Extension -in @('.dll','.exe') } | ForEach-Object {
    [ordered]@{name=$_.Name; sha256=(Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()}
})
$Manifest = [ordered]@{commit=$Commit; patch_sha256=(Get-FileHash "$Root\patches\runtime.patch" -Algorithm SHA256).Hash.ToLowerInvariant(); files=$Files}
foreach ($Name in @('runtime-built.json','runtime-built-mtp.json')) {
    $Manifest | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 "$Root\reports\$Name"
}
