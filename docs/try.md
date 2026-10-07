# Try SimulSI in your browser

This page runs SimulSI with [Pyodide](https://pyodide.org) (Python compiled
to WebAssembly): nothing is installed and nothing leaves your browser. The
first run downloads Python, NumPy, SciPy and SimulSI from public CDNs
(about 30 MB, cached afterwards), so it takes a moment.

<div id="simulsi-try" markdown="0">
  <label for="try-example"><strong>Example</strong></label>
  <select id="try-example" style="margin: 0 0 .5rem .5rem"></select>
  <textarea id="try-code" spellcheck="false" style="width:100%;height:18rem;font-family:var(--md-code-font-family,monospace);font-size:.75rem;padding:.6rem;border:1px solid #cbd5e1;border-radius:6px"></textarea>
  <p>
    <button id="try-run" class="md-button md-button--primary">Run</button>
    <span id="try-status" style="margin-left:.75rem;color:#64748b"></span>
  </p>
  <pre id="try-output" style="min-height:6rem;max-height:32rem;overflow:auto;background:#0f172a;color:#e2e8f0;padding:.8rem;border-radius:6px;font-size:.72rem;white-space:pre"></pre>
</div>

<script>
(function () {
  const EXAMPLES = {
    "Airline: what if we pad the schedule?": `from simulsi import Experiment
from simulsi.models import airline

result = Experiment(airline, airline.preset_scenarios(), replications=4, seed=1).run()
print(result.format_summary(["otp", "delay.arrival.mean", "reactionary_share"]))`,
    "Epidemic: vaccination vs lockdown": `from simulsi import Experiment, Scenario
from simulsi.models import epidemic

scenarios = [Scenario("no measures", {"lockdown_trigger": 2.0}),
             Scenario("lockdown at 80% beds", {}),
             Scenario("40% vaccinated", {"vaccination": 0.4, "lockdown_trigger": 2.0})]
res = Experiment(epidemic.with_options(duration=200), scenarios, replications=3, seed=2).run()
print(res.format_summary(["attack_rate", "people.I.max", "deaths"]))`,
    "Flowchart model in YAML": `from simulsi.flowchart import load_flowchart

clinic = load_flowchart("""
flow:
  name: clinic
  duration: 480
  parameters: {doctors: 2}
  resources: {nurse: 1, doctor: $doctors}
  sources: [{name: patient, interarrival: {distribution: exponential, mean: 6}, next: triage}]
  stations:
    triage: {resource: nurse, service: {distribution: triangular, low: 2, mode: 4, high: 7}, next: consult}
    consult: {resource: doctor, service: {distribution: exponential, mean: 10}, next: exit}
""")
for doctors in (1, 2, 3):
    wait = clinic.evaluate({"doctors": doctors}, metric="entity.patient.time_in_system.mean", replications=5)
    print(f"{doctors} doctor(s): mean time in clinic {wait:6.1f} min")`,
    "Trade-offs: Pareto front": `from simulsi.models import disruption_recovery
from simulsi.optimization import pareto_search

front = pareto_search(disruption_recovery, {"cost": "min", "otp": "max"},
                      grid={"policy": ["delay", "cancel", "spares"], "spares": [1, 4]},
                      replications=2)
print(front.format(all_designs=True))`,
    "Your own model": `from simulsi import Simulation

sim = Simulation(seed=42)
barista = sim.resource("barista", capacity=1)

def customer(sim):
    yield from sim.use(barista, sim.stream("make").exponential(2.5))

def arrivals(sim):
    while True:
        yield sim.stream("arrive").exponential(3.0)
        sim.process(customer(sim))

sim.process(arrivals(sim))
m = sim.run(until=480).metrics
print(f"utilization {m['resource.barista.utilization']:.0%}, mean wait {m['resource.barista.wait.mean']:.1f} min")`,
  };
  const PYODIDE = "https://cdn.jsdelivr.net/pyodide/v0.29.5/full/";
  const sel = document.getElementById("try-example");
  const code = document.getElementById("try-code");
  const out = document.getElementById("try-output");
  const status = document.getElementById("try-status");
  const btn = document.getElementById("try-run");
  if (!sel) return;
  Object.keys(EXAMPLES).forEach((k) => sel.add(new Option(k, k)));
  code.value = EXAMPLES[sel.value];
  sel.addEventListener("change", () => { code.value = EXAMPLES[sel.value]; out.textContent = ""; });

  let ready = null;
  function boot() {
    if (ready) return ready;
    ready = new Promise((resolve, reject) => {
      const s = document.createElement("script");
      s.src = PYODIDE + "pyodide.js";
      s.onload = async () => {
        try {
          status.textContent = "Starting Python…";
          const py = await loadPyodide({ indexURL: PYODIDE });
          status.textContent = "Installing SimulSI…";
          await py.loadPackage("micropip");
          await py.runPythonAsync("import micropip\nawait micropip.install('simulsi')");
          resolve(py);
        } catch (e) { reject(e); }
      };
      s.onerror = () => reject(new Error("could not load Pyodide from the CDN"));
      document.head.appendChild(s);
    });
    return ready;
  }

  btn.addEventListener("click", async () => {
    btn.disabled = true;
    out.textContent = "";
    const t0 = performance.now();
    try {
      const py = await boot();
      status.textContent = "Running…";
      py.setStdout({ batched: (s) => { out.textContent += s + "\n"; } });
      py.setStderr({ batched: (s) => { out.textContent += s + "\n"; } });
      await py.runPythonAsync(code.value);
      const v = py.runPython("import simulsi; simulsi.__version__");
      status.textContent = `Done in ${((performance.now() - t0) / 1000).toFixed(1)} s (SimulSI ${v})`;
    } catch (e) {
      out.textContent += String(e);
      status.textContent = "Error";
    } finally {
      btn.disabled = false;
    }
  });
})();
</script>

Runs are single-threaded in the browser and somewhat slower than native
Python; keep run lengths and replications small. Install SimulSI locally
(`pip install simulsi`) for real studies, worker processes and the
dashboard.
