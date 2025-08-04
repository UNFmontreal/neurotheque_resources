import click
import yaml
import json
from pathlib import Path
import os
from jsonschema import validate, ValidationError

def get_available_steps():
    """Get a list of available pipeline steps from the scr/steps directory."""
    steps_dir = Path(__file__).parent / 'steps'
    steps = [f.stem for f in steps_dir.glob('*.py') if f.stem not in ['__init__', 'base']]
    return steps

def main():
    """Main function to run the interactive configuration generator."""
    
    # Load the schema for validation
    schema_path = Path(__file__).parent / 'config_schema.json'
    with open(schema_path, 'r') as f:
        schema = json.load(f)

    config = {}

    click.echo("Welcome to the Neuroflow Configuration Wizard!")
    click.echo("This wizard will help you create a configuration file for your pipeline.")
    
    # --- Directory Configuration ---
    click.echo("\n--- Directory Configuration ---")
    root_dir = click.prompt("Enter the project root directory", type=click.Path(exists=True, file_okay=False, resolve_path=True), default=os.getcwd())
    config['directory'] = {
        'root': root_dir,
        'raw_data_dir': click.prompt("Enter the raw data directory (relative to root)", default='data/raw'),
        'processed_dir': click.prompt("Enter the processed data directory (relative to root)", default='data/processed'),
        'reports_dir': click.prompt("Enter the reports directory (relative to root)", default='reports'),
        'derivatives_dir': click.prompt("Enter the derivatives directory (relative to root)", default='derivatives'),
    }

    # --- Subject and Session ---
    click.echo("\n--- Subject and Session Configuration ---")
    subjects = click.prompt("Enter subject IDs (comma-separated)", default="sub-01")
    sessions = click.prompt("Enter session IDs (comma-separated)", default="ses-001")
    config['subjects'] = [s.strip() for s in subjects.split(',')]
    config['sessions'] = [s.strip() for s in sessions.split(',')]


    # --- Pipeline Steps ---
    click.echo("\n--- Pipeline Steps Configuration ---")
    available_steps = get_available_steps()
    
    selected_steps = []
    while True:
        step_name = click.prompt(
            "Choose a step to add to the pipeline (or press Enter to finish)",
            type=click.Choice(available_steps + ['']),
            show_choices=True,
            default=''
        )
        if not step_name:
            break
        
        # For now, we'll just add the step with no params.
        # A more advanced version could prompt for params based on the step's signature.
        step_config = {'name': step_name, 'params': {}}
        selected_steps.append(step_config)
        click.echo(f"Added step: {step_name}")

    config['pipeline'] = {'steps': selected_steps}
    
    # --- Final Validation and Save ---
    try:
        validate(instance=config, schema=schema)
        click.echo("\nConfiguration is valid.")
    except ValidationError as e:
        click.echo(f"\nConfiguration Error: {e.message}")
        if click.confirm("An error occurred during validation. Do you want to save anyway?"):
            pass
        else:
            click.echo("Aborting.")
            return

    output_file = click.prompt("\nEnter the name for the output YAML file", default="config.yml")
    with open(output_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    click.echo(f"\nConfiguration successfully saved to {output_file}")
    click.echo("You can now run the pipeline using: neuroflow run " + output_file)


if __name__ == '__main__':
    main()
