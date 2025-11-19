from flask import Flask, render_template, request, redirect, url_for
from pathlib import Path
import traceback
import os
import shutil
import datetime
# import model wrappers (they must expose model_main(input_path) -> dict)
# adjust names if you renamed files; these are expected to be in models/
from models.model1 import model_main as model1_main
from models.model2 import model_main as model2_main
from models.model3 import model_main as model3_main

app = Flask(__name__, static_folder="static", template_folder="templates")

# helper: ensure static plots dir exists
PROJECT_ROOT = Path(__file__).resolve().parent
STATIC_PLOTS = PROJECT_ROOT / "static" / "plots"
STATIC_PLOTS.mkdir(parents=True, exist_ok=True)


@app.route("/")
def index():
    # show simple UI
    return render_template("index.html")


# @app.route("/run_model/<model_id>", methods=["GET", "POST"])
# def run_model(model_id):
#     """
#     model_id: "1", "2", or "3"
#     Provide match_path as form field or query param (full filesystem path to a match folder).
#     """
#     match_path = request.values.get("match_path", "").strip() or None
#     if not match_path:
#         return "Error: provide match_path (e.g. /Users/.../opendata/data/matches/1925299) as form input", 400

#     try:
#         if model_id == "1":
#             result = model1_main(match_path)
#         elif model_id == "2":
#             result = model2_main(match_path)
#         elif model_id == "3":
#             result = model3_main(match_path)
#         else:
#             return "Unknown model id", 404
#     except Exception as e:
#         tb = traceback.format_exc()
#         return f"<h3>Model error</h3><pre>{tb}</pre>", 500

#     # result is expected to be a dict containing at least:
#     # { "plots": [ "plots/xxx.png", ... ], "output_folder": "/abs/path/to/output" }
#     return render_template("results.html", model_id=model_id, match_path=match_path, result=result)
@app.route("/run_model/<model_id>", methods=["GET", "POST"])
def run_model(model_id):
    """
    model_id: "1", "2", or "3"
    Provide match_path as form field or query param (full filesystem path to a match folder).
    This wrapper runs the selected model, copies its outputs to static/plots/<run_id>/,
    and returns a template that can show links/images.
    """
    match_path = request.values.get("match_path", "").strip() or None
    if not match_path:
        return "Error: provide match_path (e.g. /Users/.../opendata/data/matches/1925299) as form input", 400

    try:
        if model_id == "1":
            result = model1_main(match_path)
        elif model_id == "2":
            result = model2_main(match_path)
        elif model_id == "3":
            result = model3_main(match_path)
        else:
            return "Unknown model id", 404
    except Exception as e:
        tb = traceback.format_exc()
        return f"<h3>Model error</h3><pre>{tb}</pre>", 500

    # --- Normalize result and copy outputs into static folder so Flask can serve them ---
    # Expectation: result is a dict, ideally with "output_folder" (absolute path) and maybe "plots"
    output_folder = result.get("output_folder") if isinstance(result, dict) else None

    # If the model didn't supply an output_folder, fall back to the old default
    if not output_folder:
        # try model-specific default
        output_folder = os.path.join(os.path.dirname(__file__), "model1_output_multi")

    # ensure it exists
    if not os.path.exists(output_folder):
        # still allow rendering — show message in template
        result.setdefault("error", f"Model finished but output folder not found: {output_folder}")
        result.setdefault("plots", [])
    else:
        # create a run-specific directory in static/plots to avoid collisions
        run_id = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        dest_dir = STATIC_PLOTS / run_id
        dest_dir.mkdir(parents=True, exist_ok=True)

        # If the model returned a list of plot filenames, copy those; otherwise copy images/csvs found
        plots_list = []
        if isinstance(result.get("plots"), (list, tuple)) and result.get("plots"):
            # result['plots'] might be relative paths or filenames
            for p in result["plots"]:
                # try to resolve an absolute path for p
                abs_p = p
                if not os.path.isabs(p):
                    # first try in output_folder, then as-is
                    candidate = os.path.join(output_folder, p)
                    if os.path.exists(candidate):
                        abs_p = candidate
                    else:
                        abs_p = os.path.join(output_folder, os.path.basename(p))
                if os.path.exists(abs_p):
                    shutil.copy2(abs_p, dest_dir)
                    plots_list.append(os.path.basename(abs_p))
        else:
            # copy common output file types from the output folder (images, csv, json)
            for fname in sorted(os.listdir(output_folder)):
                if fname.lower().endswith((".png", ".jpg", ".jpeg", ".svg", ".csv", ".json", ".pdf")):
                    src = os.path.join(output_folder, fname)
                    if os.path.isfile(src):
                        shutil.copy2(src, dest_dir)
                        plots_list.append(fname)

        # Build public URLs for the copied files and store in result for the template to use
        public_urls = []
        for fname in plots_list:
            public_urls.append(url_for("static", filename=f"plots/{run_id}/{fname}"))

        # Overwrite/attach the plots list and public output_folder location in result
        result["served_plots"] = public_urls
        result["served_output_dir"] = f"static/plots/{run_id}"

    # Render the results template (your results.html should iterate over result['served_plots'])
    return render_template("results.html", model_id=model_id, match_path=match_path, result=result)


if __name__ == "__main__":
    # dev server
    app.run(host="127.0.0.1", port=5001, debug=True)
