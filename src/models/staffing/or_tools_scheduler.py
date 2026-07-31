class ORToolsShiftScheduler:
    """Scaffold only — not wired into the pipeline or prediction API.

    TODO: integrate OR-Tools (CP-SAT) constraint solver to convert the
    regression model's `required_staff` counts per department/day into
    actual shift assignments (employee-to-shift), respecting labor-law
    constraints (max hours, rest periods) and employee availability.
    """

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError("OR-Tools shift scheduling is not yet implemented")
