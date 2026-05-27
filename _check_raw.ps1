$bytes = [System.IO.File]::ReadAllBytes('d:\Soolin\SoolinOperator\test_data\5320,4600,12bit.raw')
Write-Host ('File size: {0} bytes' -f $bytes.Length)
Write-Host ''
Write-Host 'First 32 bytes (hex):'
for ($i = 0; $i -lt 32; $i++) {
    Write-Host -NoNewline ('{0:X2} ' -f $bytes[$i])
    if (($i + 1) % 16 -eq 0) { Write-Host '' }
}
Write-Host ''
Write-Host 'First 20 uint16 values (little-endian):'
for ($i = 0; $i -lt 20; $i++) {
    $val = [BitConverter]::ToUInt16($bytes, $i * 2)
    Write-Host -NoNewline ('{0} ' -f $val)
}
Write-Host ''
$max = 0
$min = 65535
$sample = [Math]::Min(100000, $bytes.Length / 2)
for ($i = 0; $i -lt $sample; $i++) {
    $val = [BitConverter]::ToUInt16($bytes, $i * 2)
    if ($val -gt $max) { $max = $val }
    if ($val -lt $min) { $min = $val }
}
Write-Host ('Sample size: {0} uint16 values' -f $sample)
Write-Host ('Max value: {0} (0x{0:X4})' -f $max)
Write-Host ('Min value: {0} (0x{0:X4})' -f $min)

# Check if MSB or LSB aligned
$expected_12bit_max = 4095
if ($max -gt $expected_12bit_max) {
    Write-Host ('Data appears MSB-aligned (max={0} > 12-bit limit={1})' -f $max, $expected_12bit_max)
    Write-Host ('After right-shift by 4: max would be {0}' -f ($max -shr 4))
} else {
    Write-Host ('Data appears LSB-aligned (max={0} <= 12-bit limit={1})' -f $max, $expected_12bit_max)
}

# Check high byte ratio
$high_count = 0
for ($i = 0; $i -lt $sample; $i++) {
    $val = [BitConverter]::ToUInt16($bytes, $i * 2)
    if (($val -shr 8) -ne 0) { $high_count++ }
}
Write-Host ('High-byte non-zero ratio: {0:F2}%' -f ($high_count * 100.0 / $sample))
