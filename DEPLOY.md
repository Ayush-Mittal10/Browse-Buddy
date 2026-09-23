# Deploying

The server is a Python process that owns a Chromium. It needs about 2 GB of
memory, a long request timeout because the WebSocket stays open for the whole
session, and low concurrency because every visitor gets a browser of their own.

These notes use Google Cloud Run. Anything that runs a container with those
three properties will do.

## Before anything is public

Two settings, neither of them optional if strangers can reach the URL.

**Limit where it can go.** An agent that will visit any address you name is an
open proxy with a nice interface. The URL policy already refuses localhost,
private ranges and the cloud metadata endpoint, but on a shared host that is the
floor, not the ceiling:

```
BROWSER_AGENT_ALLOWED_DOMAINS=wikipedia.org,news.ycombinator.com,bbc.com,python.org
```

Commas or spaces, whichever suits the tool setting it — see the note under the
deploy command for why that matters to `gcloud`.

Those four are what the built-in suggestion pills use, and none of them fight
automated browsers — a demo's first impression should not be a CAPTCHA. Add to
the list rather than replacing it, or the suggestions stop working.

Add `youtube.com` if you want "play me a video" to work. The agent finds the
video and hands the link to the visitor's own browser, which is where it can
actually be heard — see the note on sound in README.md.

**Keep the Gemini key in a project with no billing account.** Cloud Run needs
billing; the Gemini free tier needs the absence of it, and the free tier is
per-project. Linking a billing account to the project that issued your key moves
it onto the paid tier, where every call answers

```
402 RESOURCE_EXHAUSTED — Your prepayment credits are depleted.
```

They do not have to be the same project. Run the service in the billed one and
issue the key from another.

**Cap the spend.** Every step is one API call against your key. Put the demo key
in its own workspace with a hard monthly limit — the only control that cannot be
coded around. The in-app caps (steps, tasks, concurrent browsers, idle timeout)
reduce the bill; they do not bound it.

## Getting the project ready

Cloud Run needs a billing account even to stay inside the free tier, and it will
not deploy as a service account borrowed from another project.

```bash
gcloud auth login                       # as yourself, not a service account
gcloud projects create browse-buddy-demo --name="Browse Buddy"
gcloud config set project browse-buddy-demo
gcloud billing accounts list
gcloud billing projects link browse-buddy-demo --billing-account=<ID>

gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
  artifactregistry.googleapis.com secretmanager.googleapis.com

gcloud secrets create gemini-api-key --replication-policy=automatic
grep -m1 '^GEMINI_API_KEY=' .env | cut -d= -f2- | tr -d '"\'' | tr -d '\n' \
  | gcloud secrets versions add gemini-api-key --data-file=-
```

Read out of `.env` rather than typed, and piped through stdin, so the key never
reaches the shell history and cannot be mistyped. Check what actually landed
before deploying, because a wrong value here surfaces much later as
"API key not valid" and looks like a problem with the key:

```bash
gcloud secrets versions access latest --secret=gemini-api-key | wc -c
```

That should be the length of your key. If it is 15, the placeholder went in
instead of the key.

If the deploy then complains it cannot read the secret, Cloud Run's own service
account needs to be told it may:

```bash
NUM=$(gcloud projects describe browse-buddy-demo --format='value(projectNumber)')
gcloud secrets add-iam-policy-binding gemini-api-key \
  --member="serviceAccount:${NUM}-compute@developer.gserviceaccount.com" \
  --role=roles/secretmanager.secretAccessor
```

## Deploy

```bash
gcloud run deploy browse-buddy \
  --source . \
  --region asia-south1 \
  --memory 2Gi \
  --cpu 2 \
  --concurrency 2 \
  --timeout 3600 \
  --min-instances 0 \
  --allow-unauthenticated \
  --set-env-vars 'BROWSER_AGENT_ALLOWED_DOMAINS=wikipedia.org news.ycombinator.com bbc.com python.org' \
  --set-secrets 'GEMINI_API_KEY=gemini-api-key:latest'
```

Build happens on Cloud Build from the `Dockerfile`; nothing needs Docker
locally.

Note the **spaces** in the domain list. `gcloud` reads a comma as the separator
*between* environment variables, so a comma inside a value ends the pair and the
deploy fails with "Bad syntax for dict arg". The list is read with either
separator, so spaces are the path of least resistance; `^|^KEY=a,b,c` is the
alternative if you would rather keep commas.

The service name has to be lowercase with no spaces (`^[a-z]([-a-z0-9]*[a-z0-9])?$`),
so "Browse Buddy" is rejected and `browse-buddy` is not.

Three of those flags are the ones people get wrong:

| Flag | Default | Why it has to change |
|---|---|---|
| `--timeout 3600` | 300s | The WebSocket *is* the request. At the default it is cut mid-task after five minutes. |
| `--concurrency 2` | 80 | Each session launches a Chromium. Eighty on one instance takes the instance down. |
| `--memory 2Gi` | 512Mi | Chromium alone will not fit in the default. |

The key goes in Secret Manager, never in `--set-env-vars` and never in the
image — `.dockerignore` keeps `.env` out of the build context.

## Check it actually came up

```bash
curl -s "$(gcloud run services describe browse-buddy --region asia-south1 \
  --format 'value(status.url)')/healthz"
```

`{"status":"ok"}` means the instance launched a real Chromium during startup.
`503` with a `detail` means it could not, and the instance takes itself out of
rotation rather than accepting visitors it can only disappoint.

## Cold starts

With `--min-instances 0` the service costs nothing while idle, and the first
visitor after a quiet spell waits 20–30 seconds for the image to start. That
reads as broken to somebody who was sent a link.

`--min-instances 1` removes the wait and costs roughly $5–15 a month, because an
instance that is always up is always billed. For a demo you are sending to
people, pay it.

## What the free tier covers

Roughly 180k GiB-seconds a month. At 2 GiB that is ~25 hours of active time, so
about 300 five-minute sessions. Ample for a demo — but note that an *open
WebSocket keeps the instance billable*, which is what the idle timeout is for.

## The thing that may not survive the move

The agent presents as an ordinary Chromium rather than a headless one, which is
enough for most sites from a home connection. Datacentre IP ranges are rated far
more harshly, so a search engine that works from your laptop may serve a CAPTCHA
from Cloud Run.

Test it immediately after the first deploy rather than finding out from someone
else. If it happens, point the demo at sites that do not fight automation via
the allowlist above.
