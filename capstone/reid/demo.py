"""Render the vehicle ReID pipeline."""


def main():
    from utils.cli.commands import dispatch
    return dispatch('demo')


if __name__ == '__main__':
    main()
