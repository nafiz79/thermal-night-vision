# Sequential training chain with auto-resume (thesis Appendix B.4).
# Runs experiments one after another so the GPU never idles overnight, logs every
# attempt to a timestamped file, and resumes from last.pt if a run is interrupted.
# The run definitions ($runA..$runC) are filled in from the commands in Appendix B.3.

function Log($msg) { Write-Host "[$(Get-Date -f 'yyyy-MM-dd HH:mm:ss')] $msg" }

function Invoke-Run($name, $cmd, $resumeCmd, $ckpt) {
    New-Item -ItemType Directory -Force -Path "runs/thesis/logs" | Out-Null
    for ($i = 1; $i -le 8; $i++) {
        $log = "runs/thesis/logs/${name}_$(Get-Date -f yyyyMMdd_HHmmss).log"
        Log "$name : attempt $i / 8   (log -> $log)"
        if (($i -gt 1) -and (Test-Path $ckpt)) {
            & conda run -n nightfusion @resumeCmd *>> $log      # resume last.pt
        } else {
            & conda run -n nightfusion @cmd *>> $log            # fresh start
        }
        if ($LASTEXITCODE -eq 0) { Log "$name : finished"; return $true }
        Start-Sleep -Seconds 30
    }
    Log "$name : FAILED after 8 attempts"; return $false
}

$runA        = @("python", "src/train_ultralytics.py", "--weights", "yolov8n.pt", "--data", "configs/flir_clahe.yaml", "--name", "v8n_clahe_true")
$runA_resume = @("yolo", "train", "resume", "model=runs/detect/v8n_clahe_true/weights/last.pt")

$runB        = @("python", "yolov5/train.py", "--img", "640", "--batch", "16", "--epochs", "60",
                 "--data", "configs/flir.yaml", "--weights", "yolov5n.pt",
                 "--hyp", "hyp_v5n_matched.yaml", "--name", "v5n_raw_matched")
$runB_resume = @("python", "yolov5/train.py", "--resume", "runs/train/v5n_raw_matched/weights/last.pt")

$runC        = @("python", "src/train_ultralytics.py", "--weights", "yolo11n.pt", "--data", "configs/flir_clahe.yaml", "--name", "v11n_clahe_true")
$runC_resume = @("yolo", "train", "resume", "model=runs/detect/v11n_clahe_true/weights/last.pt")

Log "Checking CUDA / GPU..."
python -c "import torch; print('CUDA', torch.cuda.is_available())"

Invoke-Run "v8n_clahe_true"  $runA $runA_resume "runs/detect/v8n_clahe_true/weights/last.pt"
Invoke-Run "v5n_raw_matched" $runB $runB_resume "runs/train/v5n_raw_matched/weights/last.pt"
Invoke-Run "v11n_clahe_true" $runC $runC_resume "runs/detect/v11n_clahe_true/weights/last.pt"
