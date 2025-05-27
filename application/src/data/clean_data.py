import pandas as pd
import click


@click.command()
@click.argument("input_path", type=click.Path())
@click.argument("output_path", type=click.Path())
def clean_data(input_path: str, output_path: str):
    df = pd.read_csv(input_path)
    df = df.drop('num_purchases', axis=1)
    df = df.dropna()
    df['day'] = pd.to_datetime(df.dates).dt.day
    df['month'] = pd.to_datetime(df.dates).dt.month
    df.to_csv(output_path, index=False)
    click.echo("Success!")


if __name__ == '__main__':
    clean_data()