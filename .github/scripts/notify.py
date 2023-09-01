import sys
import json


def notify_push(github: dict) -> str:
    event = github['event']

    pusher = event['pusher']['name']
    num_commits = len(event['commits'])
    branch = event['ref'].removeprefix('refs/heads/') # FIXME
    compare = event['compare']

    text = f"""\
{pusher} [pushed]({compare}) {num_commits} commits to branch {branch}
"""
    for commit in event['commits']:
        username = commit['author']['username']
        message = commit['message']
        url = commit['url']
        text += f'\n- {username}: [{message}]({url})'
    return text


def notify_pull_request(github: dict) -> str:
    pr = github['event']['pull_request']

    user = pr['user']['login']
    url = pr['_links']['html']['href']
    label = pr['base']['label']

    text = f"""\
{user} created [a new PR]({url}) into {label}
"""
    return text


def main() -> None:
    github = json.loads(' '.join(sys.argv[1:]))

    event_name = github['event_name']
    if event_name == 'push':
        text = notify_push(github)
    elif event_name == 'pull_request':
        text = notify_pull_request(github)
    else:
        print(f'{github=}')
        raise RuntimeError(f'got an unknown {event_name=}')

    output = {
        'text': text
    }

    with open('mattermost.json', 'w') as stream:
        json.dump(output, stream)


if __name__ == "__main__":
    main()
