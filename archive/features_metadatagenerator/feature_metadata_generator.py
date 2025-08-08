import yaml
import json
import time
import re
from typing import Dict, List, Any, Tuple
from itertools import product

class FeatureMetadataGenerator:
    def __init__(self, config_path: str):
        """Initialize with YAML config file path"""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        self.current_time = int(time.time() * 1000)  # Current time in milliseconds
    
    def generate_feature_metadata(self) -> List[Dict[str, Any]]:
        """Generate list of feature metadata from config"""
        feature_metadata_list = []
        
        # Get base configuration
        feature_category = self.config['feature_category']
        feature_type = self.config['feature_type']
        created_by = self.config['created_by']
        team_name = self.config['team_name']
        data_type = self.config.get('data_type', 'FLOAT').lower()
        # Default entity mapping for bnc_uid
        source_input_mapping = self.config.get('source_input_mapping', {'uid': 'bnc_uid'})
        
        # Get condition dictionaries - now supports dict format with mappings
        condition_lists = [
            self._parse_condition_list(self.config.get('condition_name_0', ['TRUE'])),
            self._parse_condition_list(self.config.get('condition_name_1', ['TRUE'])),
            self._parse_condition_list(self.config.get('condition_name_2', ['TRUE']))
        ]
        
        # Get time conditions - now a list of time condition names
        time_cond_list = self.config['time_cond_dict']
        
        # Get column aggregation configuration
        col_agg_dict = self.config['col_agg_dict']
        
        # Generate features for each column
        for col_name, col_config in col_agg_dict.items():
            conditions = col_config['conditions']
            aggregations = col_config['aggregations']
            
            # Generate all combinations of conditions that are enabled
            condition_combinations = self._generate_condition_combinations(conditions, condition_lists)
            
            # Generate features for each combination of conditions, aggregations, and time windows
            for condition_combo, agg_func, time_key in product(
                condition_combinations, aggregations, time_cond_list
            ):
                feature_name = self._generate_feature_name(
                    team_name, feature_category, agg_func, condition_combo, col_name, time_key
                )
                
                # Generate description
                description = self._generate_description(agg_func, col_name, condition_combo, time_key)
                
                # Create feature metadata
                feature_metadata = {
                    "feature_name": f"{feature_category}:{feature_name}",
                    "feature_type": feature_type,
                    "feature_data_type": data_type,
                    "query": "",  # Will be populated by batch query generator
                    "created_time": self.current_time,
                    "updated_time": self.current_time,
                    "created_by": created_by,
                    "entities_mapping": source_input_mapping,
                    "last_updated_by": self.config.get('last_updated_by', ''),
                    "approved_by": self.config.get('approved_by', ''),
                    "status": self.config.get('status', 'CREATED'),
                    "description": description
                }
                
                feature_metadata_list.append(feature_metadata)
        
        return feature_metadata_list
    
    def _parse_condition_list(self, condition_config: List) -> List[str]:
        """Parse condition list that may contain strings or dicts"""
        parsed_conditions = []
        for item in condition_config:
            if isinstance(item, str):
                parsed_conditions.append(item)
            elif isinstance(item, dict):
                # Extract the key as the condition name
                for key in item.keys():
                    parsed_conditions.append(key)
            else:
                parsed_conditions.append(str(item))
        return parsed_conditions
    
    def _get_sql_expression_for_condition(self, condition_name: str) -> str:
        """Get SQL expression for a condition name from config"""
        # Search through all condition lists for the mapping
        condition_configs = [
            self.config.get('condition_name_0', []),
            self.config.get('condition_name_1', []),
            self.config.get('condition_name_2', [])
        ]
        
        for condition_list in condition_configs:
            for item in condition_list:
                if isinstance(item, dict) and condition_name in item:
                    return item[condition_name]
        
        # Default mappings if not found in config
        if condition_name == 'TRUE':
            return 'TRUE'
        elif condition_name == 'is_ios':
            return '(is_ios = TRUE)'
        elif condition_name == 'is_notios':
            return '(is_ios = FALSE)'
        else:
            return condition_name
    
    def _generate_condition_combinations(self, conditions: List[bool], condition_lists: List[List[str]]) -> List[Tuple[str, ...]]:
        """Generate all valid combinations of conditions"""
        combinations = []
        
        # Get available condition keys for each enabled condition list
        available_conditions = []
        for i, is_enabled in enumerate(conditions[:3]):  # Only first 3 are regular conditions
            if is_enabled and i < len(condition_lists):
                available_conditions.append(condition_lists[i])
            else:
                available_conditions.append(['TRUE'])  # Default condition
        
        # Generate all combinations
        for combo in product(*available_conditions):
            combinations.append(combo)
        
        return combinations
    
    def _generate_feature_name(self, team_name: str, feature_category: str, agg_func: str, 
                             condition_combo: Tuple[str, ...], col_name: str, time_key: str) -> str:
        """Generate feature name following the naming convention"""
        # Build condition part - filter out TRUE and use condition names as-is
        condition_parts = []
        for cond in condition_combo:
            if cond != 'TRUE':
                condition_parts.append(cond)
        
        # Build feature name parts - short format: agg_[conditions_]column_timewindow
        parts = [agg_func.lower()]
        
        # Add conditions if they exist
        parts.extend(condition_parts)
        
        # Add column and time window
        parts.extend([col_name, time_key])
        
        # Join with underscores
        feature_name = '_'.join(parts)
        
        # Ensure feature name doesn't exceed 63 characters
        if len(feature_name) > 63:
            feature_name = feature_name[:63]
        
        return feature_name
    
    def _parse_time_key_to_description(self, time_key: str) -> str:
        """Parse time key (e.g., 'l7d', 'l1h') to human readable description"""
        import re
        
        # Match pattern like l1d, l7d, l30d, l1h, etc.
        match = re.match(r'l(\d+)([hdmwy])', time_key.lower())
        if not match:
            return time_key  # Return as-is if pattern doesn't match
        
        number = match.group(1)
        unit = match.group(2)
        
        # Map units to full words
        unit_map = {
            'h': 'hour',
            'd': 'day', 
            'm': 'month',
            'w': 'week',
            'y': 'year'
        }
        
        unit_word = unit_map.get(unit, unit)
        
        # Add plural if number > 1
        if int(number) > 1:
            unit_word += 's'
        
        return f"last {number} {unit_word}"
    
    def _generate_description(self, agg_func: str, col_name: str, condition_combo: Tuple[str, ...], time_key: str) -> str:
        """Generate human-readable description for the feature"""
        # Convert time key to readable format dynamically
        time_desc = self._parse_time_key_to_description(time_key)
        
        # Build condition description
        condition_parts = [cond for cond in condition_combo if cond != 'TRUE']
        condition_desc = ' and '.join(condition_parts) if condition_parts else ''
        
        # Build description
        description_parts = [agg_func.lower(), col_name]
        if condition_desc:
            description_parts.append(f"where {condition_desc}")
        description_parts.append(time_desc)
        
        return ' '.join(description_parts)

    def generate_batch_query(self, feature_metadata_list: List[Dict[str, Any]]) -> str:
        """Generate complete batch query from feature metadata list"""
        feature_category = self.config['feature_category']
        base_category = feature_category.replace('_v1', '')
        
        # Get the raw query (which already contains WITH and prep CTEs)
        raw_query = self.config['raw_table_query'].strip()
        
        # Build feature CTE with all feature calculations
        feature_cte = self._build_feature_cte(base_category, feature_metadata_list)
        
        # Build final select
        final_select = self._build_final_select(base_category)
        
        # Combine all parts - raw_query already has WITH, so just append
        full_query = f"{raw_query}\n{feature_cte}\n{final_select}"
        
        return full_query
    
    def _build_raw_data_cte(self, base_category: str) -> str:
        """Build the raw data CTE"""
        raw_query = self.config['raw_table_query'].strip()
        
        cte = f"WITH {base_category}_daily AS (\n"
        cte += "  " + raw_query.replace('\n', '\n  ') + "\n"
        cte += ")"
        
        return cte
    
    def _build_prep_cte(self, base_category: str) -> str:
        """Build the prep CTE with processed columns"""
        cte = f", prep AS (\n"
        cte += "  SELECT \n"
        cte += "    bnc_uid\n"
        cte += "    , id \n"
        cte += "    , create_time AS create_time \n"
        cte += "    , is_ios\n"
        cte += "    , success_flag \n"
        cte += "    , input_time\n"
        cte += "    , ((input_time - create_time)/ 86400000.0 ) as days_since_logintime\n"
        
        # Add time condition columns
        time_cond_list = self.config['time_cond_dict']
        for time_key in time_cond_list:
            cte += f"    , {time_key}\n"
        
        cte += "    , partition_date\n"
        cte += f"  FROM {base_category}_daily\n"
        cte += ")"
        
        return cte
    
    def _build_feature_cte(self, base_category: str, feature_metadata_list: List[Dict[str, Any]]) -> str:
        """Build the feature CTE with all feature calculations"""
        cte = f", feat_{base_category} AS (\n"
        cte += "  SELECT\n"
        cte += "    bnc_uid\n"
        
        # Get configuration for building feature queries
        condition_lists = [
            self._parse_condition_list(self.config.get('condition_name_0', ['TRUE'])),
            self._parse_condition_list(self.config.get('condition_name_1', ['TRUE'])),
            self._parse_condition_list(self.config.get('condition_name_2', ['TRUE']))
        ]
        time_cond_list = self.config['time_cond_dict']
        col_agg_dict = self.config['col_agg_dict']
        
        # Generate feature calculations
        for col_name, col_config in col_agg_dict.items():
            conditions = col_config['conditions']
            aggregations = col_config['aggregations']
            
            condition_combinations = self._generate_condition_combinations(conditions, condition_lists)
            
            for condition_combo, agg_func, time_key in product(
                condition_combinations, aggregations, time_cond_list
            ):
                feature_name = self._generate_feature_name(
                    self.config['team_name'], self.config['feature_category'], 
                    agg_func, condition_combo, col_name, time_key
                )
                
                # Build condition clauses - only include non-TRUE conditions
                condition_clauses = []
                for i, cond_name in enumerate(condition_combo):
                    if cond_name != 'TRUE':
                        # Get SQL expression from config mapping
                        sql_expr = self._get_sql_expression_for_condition(cond_name)
                        condition_clauses.append(sql_expr)
                
                # Build time condition - simplified to just the column name
                time_condition = time_key
                
                # Build the CASE WHEN clause - only include actual conditions
                all_conditions = condition_clauses + [time_condition]
                case_condition = " AND ".join(all_conditions)
                case_clause = f"CASE WHEN {case_condition} THEN {col_name} ELSE NULL END"
                
                # Build the aggregation
                if agg_func == 'COUNT':
                    agg_clause = f"COUNT({case_clause})"
                elif agg_func == 'MIN':
                    agg_clause = f"MIN({case_clause})"
                elif agg_func == 'MAX':
                    agg_clause = f"MAX({case_clause})"
                elif agg_func == 'AVG':
                    agg_clause = f"AVG({case_clause})"
                elif agg_func == 'STDDEV':
                    agg_clause = f"STDDEV({case_clause})"
                
                # Build the full feature line - let NULL values remain NULL
                feature_line = f"    , {agg_clause} AS {feature_name}\n"
                cte += feature_line
        
        cte += "    , partition_date\n"
        cte += "  FROM prep\n"
        cte += "  GROUP BY bnc_uid\n"
        cte += "         , partition_date\n"
        cte += ")"
        
        return cte
    
    def _build_final_select(self, base_category: str) -> str:
        """Build the final SELECT statement"""
        final_select = "SELECT \n"
        final_select += "  *\n"
        final_select += "  , DATE('${%Y-%m-%d,-0}') AS etl_date\n"
        final_select += f"FROM feat_{base_category}\n"
        final_select += "DISTRIBUTE BY partition_date;"
        
        return final_select

    def validate_feature_names(self, batch_query: str) -> Dict[str, Any]:
        """Extract feature names from batch query and validate their lengths"""
        # Pattern to match feature names in SQL (AS clause)
        # Looks for "AS feature_name" pattern
        pattern = r'\bAS\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:,|\n|$)'
        
        matches = re.findall(pattern, batch_query, re.IGNORECASE | re.MULTILINE)
        
        # Filter out non-feature columns 
        system_columns = {'etl_date', 'partition_date', 'bnc_uid', 'login_unix_ts', 'device_type', 
                         'is_ios', 'login_id', 'partition_date_unix_ts', 'days_since_logintime',
                         'l1h', 'l6h', 'l1d', 'l3d', 'l7d', 'l14d', 'l30d', 'l60d', 'l90d'}
        
        # Feature names now start with aggregation functions
        agg_functions = ['count_', 'min_', 'max_', 'avg_', 'stddev_']
        feature_names = [name for name in matches if name not in system_columns and 
                        any(name.startswith(agg) for agg in agg_functions)]
        
        # Calculate lengths and find violations
        feature_lengths = [(name, len(name)) for name in feature_names]
        violations = [(name, length) for name, length in feature_lengths if length > 63]
        
        # Statistics
        max_length = max([length for _, length in feature_lengths]) if feature_lengths else 0
        min_length = min([length for _, length in feature_lengths]) if feature_lengths else 0
        avg_length = sum([length for _, length in feature_lengths]) / len(feature_lengths) if feature_lengths else 0
        
        return {
            'total_features': len(feature_names),
            'feature_names': feature_names,
            'feature_lengths': feature_lengths,
            'violations': violations,
            'max_length': max_length,
            'min_length': min_length,
            'avg_length': round(avg_length, 2),
            'longest_feature': max(feature_lengths, key=lambda x: x[1]) if feature_lengths else None,
            'shortest_feature': min(feature_lengths, key=lambda x: x[1]) if feature_lengths else None
        }

    def generate_and_validate_features(self) -> Dict[str, Any]:
        """Generate features and validate their names in one go"""
        # Generate metadata and query
        feature_metadata = self.generate_feature_metadata()
        batch_query = self.generate_batch_query(feature_metadata)
        
        # Validate feature names
        validation_results = self.validate_feature_names(batch_query)
        
        # Add additional info
        validation_results['batch_query'] = batch_query
        validation_results['metadata_count'] = len(feature_metadata)
        
        return validation_results

# Example usage functions
def load_config_and_generate_metadata(config_path: str) -> List[Dict[str, Any]]:
    """Load config and generate feature metadata"""
    generator = FeatureMetadataGenerator(config_path)
    return generator.generate_feature_metadata()

def load_config_and_generate_query(config_path: str) -> str:
    """Load config and generate batch query"""
    generator = FeatureMetadataGenerator(config_path)
    feature_metadata = generator.generate_feature_metadata()
    return generator.generate_batch_query(feature_metadata)

def save_metadata_to_json(metadata_list: List[Dict[str, Any]], output_path: str):
    """Save metadata list to JSON file"""
    with open(output_path, 'w') as f:
        json.dump(metadata_list, f, indent=2)

def save_query_to_sql(query: str, output_path: str):
    """Save query to SQL file"""
    with open(output_path, 'w') as f:
        f.write(query)

# Example usage
if __name__ == "__main__":
    # Load configuration and generate metadata
    config_file = "login_history_pc1_config.yaml"
    
    # Generate feature metadata
    metadata = load_config_and_generate_metadata(config_file)
    save_metadata_to_json(metadata, "feature_metadata.json")
    
    print(f"Generated {len(metadata)} features")
    print("Sample feature metadata:")
    print(json.dumps(metadata[0], indent=2))
    
    # Generate batch query
    batch_query = load_config_and_generate_query(config_file)
    save_query_to_sql(batch_query, "full_batch_query.sql")
    
    print("\nBatch query generated successfully!")
    print("First 500 characters of query:")
    print(batch_query[:500] + "..." if len(batch_query) > 500 else batch_query)
    
    # Validate feature names
    generator = FeatureMetadataGenerator(config_file)
    validation_results = generator.generate_and_validate_features()
    
    print(f"\n--- Feature Name Validation ---")
    print(f"Total features: {validation_results['total_features']}")
    print(f"Max length: {validation_results['max_length']} characters")
    print(f"Min length: {validation_results['min_length']} characters")
    print(f"Average length: {validation_results['avg_length']} characters")
    
    if validation_results['violations']:
        print(f"❌ VIOLATIONS: {len(validation_results['violations'])} features exceed 63 characters")
        for name, length in validation_results['violations']:
            print(f"  - '{name}' ({length} chars)")
    else:
        print("✅ ALL FEATURES PASS: No features exceed 63 characters")