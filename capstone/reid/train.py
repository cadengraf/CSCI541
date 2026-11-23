"""Train the joint model or specialist; run with --help for current commands."""
from utils.cli.config import CONFIG, ROOT, demo_checkpoint, path_from_config, read_config


def main():
    from utils.cli.commands import dispatch
    return dispatch('train')


if __name__ == '__main__':
    main()
