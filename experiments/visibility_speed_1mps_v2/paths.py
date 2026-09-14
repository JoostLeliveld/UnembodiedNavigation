from pathlib import Path
import os
import sys
HERE=Path(__file__).resolve().parent
R=HERE.parents[1]
O=Path(os.environ.get('BAYESIAN_NAVIGATION_ROOT', R/'logs/bayesian_navigation'))
T=O/'visibility_speed_1mps_v2_reporting'
sys.path[:0]=[str(HERE),str(O),str(O/'final_model'),str(O/'planning_joint'),str(R/'src/planning'),str(R/'src/unav_common'),str(R/'src/reliability')]
