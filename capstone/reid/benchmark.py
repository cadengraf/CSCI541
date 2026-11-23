"""Evaluate vehicle identities or rebuild reports."""


def main():
    from utils.cli.commands import dispatch
    return dispatch('benchmark')


if __name__ == '__main__':
    main()
