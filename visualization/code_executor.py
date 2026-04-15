import traceback


# =========================
# Execute generated code
# =========================
def execute_code(code):
    if not code:
        return None

    try:
        # Create a safe execution environment
        local_vars = {}

        # Execute the generated code
        exec(code, {}, local_vars)

        # Expect a Plotly figure named 'fig'
        fig = local_vars.get("fig", None)

        if fig is None:
            print(" No 'fig' found in generated code")
            return None

        return fig

    except Exception as e:
        print(" Execution error:")
        traceback.print_exc()
        return None