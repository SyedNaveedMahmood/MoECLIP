# MoECLIP ablation suite (resumable): train on VisA, evaluate per Table 2 protocol.
# train.py auto-resumes from ckpt/<name>/moe_last.pth if present.
$py = "C:\Users\user7\miniconda3\envs\moeclip\python.exe"
Set-Location "C:\Users\user7\Desktop\moeclip"
$evalSets = @("MVTec", "DTD-Synthetic", "headct", "Colon_colonDB")

$variants = @(
    @{ name = "ablation_noetf";  args = @("--etf_loss_lambda", "0") },
    @{ name = "ablation_nopaa";  args = @("--no_use_paa") },
    @{ name = "experts_k1";      args = @("--moe_num_experts", "1") },
    @{ name = "experts_k2";      args = @("--moe_num_experts", "2") },
    @{ name = "experts_k8";      args = @("--moe_num_experts", "8") }
)

foreach ($v in $variants) {
    $sp = "ckpt/$($v.name)"
    $doneMarker = "ckpt/$($v.name)/.evals_done"
    if (Test-Path $doneMarker) { Write-Output "===== SKIP $($v.name) (done) ====="; continue }
    Write-Output "===== TRAIN $($v.name) ====="
    & $py train.py --dataset VisA --save_path $sp @($v.args) *>> "$($v.name)_log.txt"
    foreach ($d in $evalSets) {
        Write-Output "===== EVAL $($v.name) / $d ====="
        & $py test.py --dataset $d --save_path $sp --eval_epoch 20 *>> "$($v.name)_results.txt"
    }
    New-Item -Path $doneMarker -ItemType File -Force | Out-Null
}
Write-Output "===== ALL ABLATIONS DONE ====="