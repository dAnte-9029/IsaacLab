from scripts.flapping_rl.pretrain_path_tracking_bc import build_bc_parser


def test_bc_parser_accepts_task_and_dataset() -> None:
    parser = build_bc_parser()
    args = parser.parse_args(
        [
            "--task",
            "Isaac-FlappingBot-PathTracking-DeLaurier-Direct-v0",
            "--dataset",
            "tmp.pt",
        ]
    )
    assert args.task.endswith("PathTracking-DeLaurier-Direct-v0")
    assert args.dataset == "tmp.pt"
