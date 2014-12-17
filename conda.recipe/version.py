from __future__ import print_function

import os
import subprocess
import sys


def version_str(directory, branch, SHA):
    """
    Get the version of biggus for a given SHA on a given branch.
    """
    cmd = ['git', 'log' , '--oneline', SHA]
    commits = subprocess.check_output(cmd, cwd=directory).strip()
    if commits:
        n_commits = len(commits.split(b'\n'))
    else:
        n_commits = 0
    return 'test.{branch}.{n_commits}.g{SHA}'.format(branch=branch,
                                               n_commits=n_commits,
                                               SHA=SHA)


def compute_version(git_directory):
    if os.environ.get('TRAVIS_TAG', ''):
        print('Using TRAVIS_TAG as version string.', file=sys.stderr)
        version = os.environ['TRAVIS_TAG']
    else:
        if os.environ.get('TRAVIS_BRANCH', ''):
            print('Using TRAVIS_BRANCH to compute version string.', file=sys.stderr)
            branch = os.environ['TRAVIS_BRANCH']
        else:
            print('Attempting to determine the branch of the git repo - '
                  'this is not 100% fool-proof.', file=sys.stderr)
            cmd = ['git', 'rev-parse', '--abbrev-ref', 'HEAD']
            branch = subprocess.check_output(cmd, cwd=git_directory).strip().decode('ascii')
        sha = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip().decode('ascii')
        version = version_str(git_directory, branch, sha)
        print('Computed version as:', version, file=sys.stderr)

    return version


if __name__ == '__main__':
    print(compute_version(os.getcwd()))
