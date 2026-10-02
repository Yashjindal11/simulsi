# SimulSI dashboard

An optional React + TypeScript + Vite + Tailwind CSS front end for
`simulsi ui`. The Python package never needs it; it only reads the local
JSON API served by `simulsi/web/server.py`.

```bash
cd web/frontend
npm ci
npm run build        # type-checks, then writes ../../src/simulsi/web/static
simulsi ui results/my-study
```

For development with hot reload, run the API and Vite side by side:

```bash
simulsi ui results/my-study --no-browser   # API on 127.0.0.1:8642
cd web/frontend && npm run dev              # UI on 127.0.0.1:5173, proxies /api
```

Charts are small hand-written SVG components (`src/charts.tsx`), so the only
runtime dependencies are React and React DOM.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/info` | version and runnable models |
| GET | `/api/models` | model descriptions and parameter schemas |
| GET | `/api/results` | loaded experiments |
| GET | `/api/results/{id}` | metadata, scenarios, parameters, summary with CIs |
| GET | `/api/results/{id}/metric?metric=&scenario=` | per-replication values, summary, convergence |
| GET | `/api/results/{id}/compare?baseline=&metrics=` | differences vs baseline with CIs |
| GET | `/api/results/{id}/export.json` / `export.csv` | downloads |
| POST | `/api/run` | run an experiment `{model, scenarios, replications, seed, duration?}` |
| POST | `/api/trace` | one traced run `{model, parameters, seed, duration?}` |
| POST | `/api/results` | register an `experiment.json` loaded in the browser |

The server binds to 127.0.0.1, checks `Host`/`Origin` headers, accepts only
JSON POST bodies (20 MB maximum), caps experiment size, and runs only
built-in models or models passed with `--model`.
