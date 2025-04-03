class BenchmarkArgs:
    def __init__(self, config_file, mode, checklist_path, device):
        self.config_file = config_file
        self.mode = mode
        self.checklist_path = checklist_path
        self.device = device

    @classmethod
    def from_parser(cls, args):
        return cls(
            config_file=args.config_file,
            mode=args.mode,
            checklist_path=args.checklist_path,
            device=args.device
        )