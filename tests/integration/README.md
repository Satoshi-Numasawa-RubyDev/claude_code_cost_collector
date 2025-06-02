# Integration Tests for Claude Code Cost Collector

This directory contains comprehensive integration tests that verify the end-to-end functionality of the Claude Code Cost Collector application.

## Test Structure

### Test Files

1. **test_end_to_end.py** - Core end-to-end functionality tests
   - Basic output format tests (text, JSON, YAML, CSV)
   - Aggregation granularity tests (daily, monthly, project, session, all)
   - Date filtering tests
   - Error handling tests
   - Combined feature tests

2. **test_output_validation.py** - Output accuracy and consistency tests
   - Data consistency across formats
   - Cost calculation accuracy
   - Token count accuracy
   - Aggregation accuracy
   - Format compliance tests

3. **test_error_handling.py** - Comprehensive error scenario tests
   - Invalid input handling
   - File system error handling
   - Malformed data handling
   - Resource constraint tests

### Test Data

The `test_data/` directory contains:
- **project1/** and **project2/** - Sample log files organized by date
- **empty_dir/** - Empty directory for testing
- **invalid.json** - Malformed JSON for error testing
- **missing_fields.json** - JSON with missing required fields

#### Sample Data Structure
```
test_data/
├── project1/
│   ├── 2025-05-09/session1.json  ($0.045, 1300 tokens)
│   └── 2025-05-10/session2.json  ($0.075, 2000 tokens)
├── project2/
│   ├── 2025-05-15/session3.json  ($0.032, 1000 tokens)
│   └── 2025-06-01/session4.json  ($0.089, 2600 tokens)
├── empty_dir/
├── invalid.json
└── missing_fields.json
```

**Total Expected Values:**
- Total Cost: $0.241 USD
- Total Tokens: 6,900
- Number of entries: 4
- Date range: 2025-05-09 to 2025-06-01

## Running Tests

### Run All Integration Tests
```bash
uv run python -m pytest tests/integration/ -v
```

### Run Specific Test File
```bash
uv run python -m pytest tests/integration/test_end_to_end.py -v
```

### Run Specific Test
```bash
uv run python -m pytest tests/integration/test_end_to_end.py::TestEndToEndIntegration::test_basic_text_output -v
```

## Test Categories

### 1. Output Format Tests
- Verify that all output formats (text, JSON, YAML, CSV) work correctly
- Validate format compliance and structure
- Test data consistency across formats

### 2. Aggregation Tests
- Daily aggregation by date
- Monthly aggregation by month
- Project aggregation by project directory
- Session aggregation by session ID
- Individual entry display (all)

### 3. Date Filtering Tests
- Start date filtering
- End date filtering
- Date range filtering
- Edge cases with no matching dates


### 4. Error Handling Tests
- Invalid directories and file paths
- Malformed JSON files
- Missing required fields
- Invalid command-line arguments
- Large data handling
- Memory constraints

### 6. Data Accuracy Tests
- Cost calculation verification
- Token count verification
- Aggregation math verification
- Date filtering accuracy

## Expected Test Behavior

### Success Cases (Return Code 0)
- Valid operations with proper data
- Empty directories (with informative messages)
- Exchange rate failures (with fallback to USD)

### Argument Validation Errors (Return Code 2)
- Invalid directory paths
- Invalid date formats
- Invalid command-line options

### Runtime Errors (Return Code 1)
- Unexpected exceptions during execution
- Critical failures that prevent completion

## Test Data Maintenance

When modifying test data:

1. Update expected values in test cases
2. Ensure data represents realistic log entries
3. Maintain variety in:
   - Dates (different days/months)
   - Projects (different directories)
   - Models (different Claude models)
   - Token counts and costs

## Integration with CI/CD

These tests are designed to:
- Run quickly (< 10 seconds total)
- Be deterministic and reliable
- Not require external dependencies (uses mocking for APIs)
- Provide comprehensive coverage of user scenarios

## Test Coverage

The integration tests cover:
- All major user workflows
- All output formats and aggregation methods
- Error conditions users might encounter
- Edge cases and boundary conditions
- Performance with reasonable data volumes

For unit test coverage, see the `tests/` directory at the project root.