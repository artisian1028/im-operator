$bytes = [System.IO.File]::ReadAllBytes('d:\Soolin\SoolinOperator\test_data\5320,4600,12bit.raw')

Write-Host "=== Test file analysis ==="
Write-Host ('Size: {0} bytes = {0} * 2 = {1} pixels' -f $bytes.Length, ($bytes.Length / 2))
Write-Host ('5320 * 4600 = {0}' -f (5320 * 4600))
Write-Host ''

# Simulate detect_bit_depth() logic
$byteSize = $bytes.Length

# Step 1: match_common_resolution(byte_size)
$common = @(640*480, 720*480, 720*576, 800*600, 1024*768, 1280*720, 1280*800, 1280*1024, 1440*900, 1920*1080, 1920*1200, 2048*1536, 2560*1440, 2560*1600, 3840*2160, 4096*2160, 4096*3072, 5120*2880, 5320*4600, 7680*4320, 1006*758, 4024*3032)
Write-Host "Step 1: byte_size matches known resolution? $($common -contains $byteSize)"

# Step 2: byte_size/2 matches?
$halfSize = $byteSize / 2
Write-Host "Step 2: byte_size/2 = $halfSize matches known resolution? $($common -contains $halfSize)"
Write-Host "  5320*4600 = $(5320*4600)"

# Simulate guess_bit_from_16bit_data
$max = 0
$sample = [Math]::Min(100000, $byteSize / 2)
for ($i = 0; $i -lt $sample; $i++) {
    $val = [BitConverter]::ToUInt16($bytes, $i * 2)
    if ($val -gt $max) { $max = $val }
}
Write-Host ''
Write-Host "guess_bit_from_16bit_data: max_val = $max"
if ($max -le 255) { Write-Host "  -> bit_depth = 8" }
elseif ($max -le 1023) { Write-Host "  -> bit_depth = 10" }
elseif ($max -le 4095) { Write-Host "  -> bit_depth = 12" }
elseif ($max -le 16383) { Write-Host "  -> bit_depth = 14" }
else { Write-Host "  -> bit_depth = 16" }
Write-Host "  BUT actual bit_depth should be 12 (MSB-aligned)"

# Simulate align_raw_value
Write-Host ''
Write-Host "=== align_raw_value() simulation ==="
$safeMax = (1 -shl 12) - 1
Write-Host "safe_max_val(12) = $safeMax"
$alignedMax = if ($max -gt $safeMax) { $max -shr (16 - 12) } else { $max }
Write-Host "After right-shift by 4: max = $alignedMax (should be <= 4095)"

# What would happen WITHOUT align_raw_value()?
Write-Host ''
Write-Host "=== WITHOUT align_raw_value(): pure white analysis ==="
Write-Host "All raw values are in range [6112, 37072]"
Write-Host "ALL values > safe_max_val(12) = 4095"
Write-Host "If algorithms clamp to max_val=4095: ALL pixels become 4095"
Write-Host "In save_bmp(): (4095 * 255 + 2047) / 4095 = 255 -> PURE WHITE!"
Write-Host ""

# What happens WITH align_raw_value()?
Write-Host "=== WITH align_raw_value(): correct ==="
Write-Host "Raw values [6112, 37072] >> 4 = [382, 2317]"
Write-Host "All values <= 4095, no clamping needed"
Write-Host "save_bmp(): expected_max=4095, actual_max~=2317, is_msb_aligned=false"
$maxBmp = [Math]::Floor(($alignedMax * 255 + 2047) / 4095)
Write-Host "Max BMP pixel: ($alignedMax * 255 + 2047) / 4095 = $maxBmp / 255"
Write-Host "Image will be correctly exposed (not pure white)"
