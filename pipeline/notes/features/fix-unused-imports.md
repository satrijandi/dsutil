# Fix Unused Imports

## Task
Clean up unused imports in feature_engineering.py identified by Pylance:
- `as_completed` from concurrent.futures (line 6) - not used
- `partial` from functools (line 7) - not used  
- `month` parameter in `_calculate_feature_drift` function (line 65) - not accessed

## Analysis
These imports were likely added during development but are no longer needed:
1. `as_completed` - the code uses `executor.map()` instead of `as_completed()`
2. `partial` - no partial function application is being used
3. `month` parameter - only used in a log message string conversion, can be removed

## Changes Made
- Remove unused imports from concurrent.futures and functools
- Remove unused `month` parameter from `_calculate_feature_drift` function
- Update function signature and calls accordingly