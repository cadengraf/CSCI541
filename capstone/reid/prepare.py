"""Prepare vehicle datasets."""


def main():
    from utils.cli.commands import dispatch
    return dispatch('prepare')


if __name__ == '__main__':
    main()
