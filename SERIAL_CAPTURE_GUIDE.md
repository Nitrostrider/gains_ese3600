# Serial Monitor Output Capture Guide

This guide shows you how to capture serial monitor output to a text file for debugging and analysis.

## Method 1: Using the Capture Script (RECOMMENDED)

The easiest way to capture serial output with timestamps:

```bash
# Make script executable (first time only)
chmod +x capture_serial.sh

# Run the capture script (auto-generates filename with timestamp)
./capture_serial.sh

# Or specify a custom filename
./capture_serial.sh my_test_output.txt
```

**Features:**
- ✅ Adds timestamps to each line
- ✅ Displays output in terminal AND saves to file
- ✅ Auto-creates `logs/` directory
- ✅ Default filename includes date/time

**Output location:** `logs/serial_output_YYYYMMDD_HHMMSS.txt`

**To stop capturing:** Press `Ctrl+C`

---

## Method 2: Simple Redirection

Capture output without timestamps:

```bash
# Capture to file only (no terminal display)
pio device monitor > logs/output.txt

# Capture to file AND display in terminal
pio device monitor | tee logs/output.txt
```

---

## Method 3: Capture with Filtering

Only save lines containing specific keywords:

```bash
# Only capture lines with [PROCESSED] or [RAW]
pio device monitor | grep -E "\[PROCESSED\]|\[RAW\]" | tee logs/debug_output.txt

# Only capture inference results
pio device monitor | grep -E "PREDICTION|Posture:" | tee logs/predictions.txt
```

---

## Method 4: Using `screen` with Logging

Alternative using the `screen` utility:

```bash
# Start screen with logging enabled
screen -L -Logfile logs/screen_output.txt /dev/cu.usbserial-* 115200

# Press Ctrl+A then D to detach
# Press Ctrl+A then K to kill
```

---

## Understanding the Debug Output

When you run the code with debug enabled, you'll see this every 0.5 seconds:

```
[RAW] ax=0.012, ay=-0.005, az=0.982 | gx=0.234, gy=-0.156, gz=0.089
[PROCESSED] ax=0.008, ay=-0.003, az=0.021 | gx=0.012, gy=-0.008, gz=0.005
```

**What to look for:**

### Raw Data (from sensor):
- `ax, ay`: Should be small when level (< ±0.2g)
- `az`: Should be ~1.0g when horizontal (gravity pointing down)
- `gx, gy, gz`: Angular velocity, should be ~0 when still

### Processed Data (after filters):
- `ax, ay, az`: **Linear acceleration only** (gravity removed)
  - Should be **near 0** when still (< ±0.05g)
  - Should **increase** during movement
  - `az` should **NOT** be 1.0g (that was gravity, now removed!)
- `gx, gy, gz`: **Drift-free** angular velocity
  - Should be near 0 when still (< ±1 deg/s)
  - Should increase during rotation

---

## Debugging Checklist

### ✅ Preprocessing is working if:
1. **RAW az ≈ 1.0g** when device is horizontal
2. **PROCESSED az ≈ 0.0g** when device is horizontal and still
3. **PROCESSED values change** when you move the device
4. **No filter initialization errors** in startup logs

### ❌ Preprocessing NOT working if:
1. PROCESSED az still ~1.0g (gravity not removed)
2. PROCESSED values are all exactly 0.0 (filter not running)
3. PROCESSED values identical to RAW (filter bypassed)
4. See "ERROR" messages about preprocessing

---

## Example Testing Session

1. **Upload code and start capture:**
   ```bash
   pio run -t upload
   ./capture_serial.sh test_run_1.txt
   ```

2. **Wait for initialization:**
   - Look for "✓ Preprocessing filters initialized"
   - Look for "System ready!"

3. **Press button to start inference**
   - LED should turn on
   - Should see "Started recording" on OLED

4. **Perform test movements:**
   - Stay still for 5 seconds (check PROCESSED values near 0)
   - Do a push-up slowly
   - Check predictions appear every 1 second

5. **Stop capture:** Press `Ctrl+C`

6. **Review output:**
   ```bash
   cat logs/test_run_1.txt | grep -E "\[PROCESSED\]" | head -20
   ```

---

## Analyzing Captured Data

### Extract only predictions:
```bash
grep "Posture:" logs/output.txt > logs/predictions_only.txt
```

### Count each prediction type:
```bash
grep "Posture:" logs/output.txt | cut -d: -f2 | sort | uniq -c
```

### View last 50 processed samples:
```bash
tail -100 logs/output.txt | grep "\[PROCESSED\]"
```

### Check for errors:
```bash
grep -i "error\|failed\|warning" logs/output.txt
```

---

## Troubleshooting

### "Permission denied" when running script
```bash
chmod +x capture_serial.sh
```

### "Device not found" error
```bash
# List available serial ports
ls /dev/cu.*

# If device is on different port, edit platformio.ini
# Or specify manually:
screen /dev/cu.usbserial-XXXXXX 115200
```

### Output file is empty
- Make sure device is connected
- Check USB cable is data-capable (not charge-only)
- Verify baud rate is 115200

### Too much output / file too large
```bash
# Limit file size to 10MB
pio device monitor | head -c 10M > logs/output.txt

# Or capture for limited time (60 seconds)
timeout 60 pio device monitor > logs/output.txt
```

---

## Tips

1. **Always capture during testing** - Serial output is your best debugging tool
2. **Use descriptive filenames** - e.g., `test_good_form.txt`, `test_hips_high.txt`
3. **Keep logs organized** - All files go to `logs/` directory automatically
4. **Compare captures** - Diff two files to see changes between code versions
5. **Share logs** - Easy to send log files when asking for help

---

## Quick Reference

| Task | Command |
|------|---------|
| Capture with timestamps | `./capture_serial.sh` |
| Capture without timestamps | `pio device monitor \| tee logs/output.txt` |
| View live + save | `pio device monitor \| tee logs/output.txt` |
| Only PROCESSED lines | `pio device monitor \| grep PROCESSED \| tee logs/debug.txt` |
| Stop capturing | `Ctrl+C` |
| List saved logs | `ls -lh logs/` |
| View last 20 lines | `tail -20 logs/output.txt` |
