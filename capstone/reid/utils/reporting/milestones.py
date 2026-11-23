"""Refresh the single best-model scorecard from saved evidence; no model jobs."""

def run():
    from utils.benchmarking.suite.command import main
    return main(['--report-only'])


if __name__ == '__main__':
    run()
