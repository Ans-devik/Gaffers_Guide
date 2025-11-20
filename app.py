from flask import Flask, render_template, request, redirect, url_for
from pathlib import Path
import traceback
import os
import shutil
import datetime
import logging

# import model wrappers (they must expose model_main(input_path) -> dict or string)
from models.model1 import model_main as model1_main
from models.model2 import model_main as model2_main
from models.model3 import model_main as model3_main

# --- config ---
ALLOWED_COPY_EXT = (".png", ".jpg", ".jpeg", ".svg", ".pdf", ".csv", ".json")
MAX_FILES_TO_COPY = 200  # safety cap per run

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="static", template_folder="templates")

# helper: ensure static plots dir exists
PROJECT_ROOT = Path(__file__).resolve().parent
STATIC_PLOTS = PROJECT_ROOT / "static" / "plots"
STATIC_PLOTS.mkdir(parents=True, exist_ok=True)


def ensure_result_dict(result):
    """
    Ensure we always work with a dict. Convert non-dict results into {'result': str(...)}.
    Also preserve dicts as-is.
    """
    if isinstance(result, dict):
        return result
    return {"result": str(result)}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/run_model/<model_id>", methods=["GET", "POST"])
def run_model(model_id):
    """
    Runs a chosen model by id with a provided match_path (query or form).
    Copies model outputs into static/plots/<run_id>/ and returns a template that shows images.
    """
    match_path = request.values.get("match_path", "").strip() or None
    if not match_path:
        return "Error: provide match_path (e.g. /Users/.../opendata/data/matches/1925299) as form input", 400

    # call model
    try:
        if model_id == "1":
            raw_result = model1_main(match_path)
        elif model_id == "2":
            raw_result = model2_main(match_path)
        elif model_id == "3":
            raw_result = model3_main(match_path)
        else:
            return "Unknown model id", 404
    except Exception:
        tb = traceback.format_exc()
        logger.exception("Model raised an exception")
        # For dev it's useful to return traceback; in prod consider hiding it.
        return f"<h3>Model error</h3><pre>{tb}</pre>", 500

    # Normalize result to dict
    result = ensure_result_dict(raw_result)

    # Try to discover output folder reported by the model
    output_folder = result.get("output_folder")
    if not output_folder:
        # fallback: try model-specific default names (if present), else None
        candidate_defaults = [
            os.path.join(os.path.dirname(__file__), f"model{model_id}_output_multi"),
            os.path.join(os.path.dirname(__file__), "model1_output_multi"),
            os.path.join(os.path.dirname(__file__), "output"),
            os.path.join(os.path.dirname(__file__), "outputs"),
        ]
        output_folder = next((p for p in candidate_defaults if os.path.exists(p)), None)

    # If we still don't have an output folder, record a helpful note and set empty plots list
    if not output_folder or not os.path.exists(output_folder):
        msg = f"Model finished but output folder not found: {output_folder}"
        logger.warning(msg)
        # ensure result dict has helpful fields
        result.setdefault("error", msg)
        result.setdefault("plots", [])
        result.setdefault("served_plots", [])
        result.setdefault("served_output_dir", None)
        return render_template("results.html", model_id=model_id, match_path=match_path, result=result)

    # Create a run-specific directory in static/plots to avoid collisions
    run_id = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    dest_dir = STATIC_PLOTS / run_id
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Gather files to copy
    plots_list = []
    try:
        # If model returned explicit 'plots' list, prefer those
        reported = result.get("plots") or []
        if isinstance(reported, (list, tuple)) and reported:
            for p in reported:
                if not p:
                    continue
                abs_p = p if os.path.isabs(p) else os.path.join(output_folder, p)
                # final fallback: try basename in output_folder
                if not os.path.exists(abs_p):
                    abs_p = os.path.join(output_folder, os.path.basename(p))
                if os.path.exists(abs_p) and os.path.isfile(abs_p):
                    try:
                        shutil.copy2(abs_p, dest_dir)
                        plots_list.append(os.path.basename(abs_p))
                        if len(plots_list) >= MAX_FILES_TO_COPY:
                            break
                    except Exception as e:
                        logger.warning("Failed to copy %s -> %s : %s", abs_p, dest_dir, e)
        else:
            # discover files in output_folder
            for fname in sorted(os.listdir(output_folder)):
                if len(plots_list) >= MAX_FILES_TO_COPY:
                    break
                if fname.lower().endswith(ALLOWED_COPY_EXT):
                    src = os.path.join(output_folder, fname)
                    if os.path.isfile(src):
                        try:
                            shutil.copy2(src, dest_dir)
                            plots_list.append(fname)
                        except Exception as e:
                            logger.warning("Failed to copy %s -> %s : %s", src, dest_dir, e)
    except Exception as e:
        logger.exception("Error while collecting/copying output files: %s", e)

    # Build public URLs
    public_urls = [url_for("static", filename=f"plots/{run_id}/{fname}") for fname in plots_list]

    # Attach served info to result
    result["served_plots"] = public_urls
    result["served_output_dir"] = f"static/plots/{run_id}"

    # Render the results template (your results.html should iterate over result['served_plots'])
    return render_template("results.html", model_id=model_id, match_path=match_path, result=result)


if __name__ == "__main__":
    # dev server
    app.run(host="127.0.0.1", port=5001, debug=True)
