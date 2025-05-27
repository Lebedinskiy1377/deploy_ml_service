import click
import pandas as pd


@click.command()
@click.argument('first_group_sku', type=click.Path(exists=True))
@click.argument('second_group_sku', type=click.Path(exists=True))
@click.argument('third_group_sku', type=click.Path(exists=True))
@click.argument('promo', type=click.Path(exists=True))
@click.argument('sku_dict', type=click.Path(exists=True))
@click.argument('output', type=click.Path())
def merge_data(output: str,
               first_group_sku: str,
               second_group_sku: str,
               third_group_sku: str,
               promo: str,
               sku_dict: str):
    """
    merged dataset for SKU price forecast.
    """
    first_group_sku = pd.read_csv(first_group_sku)
    second_group_sku = pd.read_csv(second_group_sku)
    third_group_sku = pd.read_csv(third_group_sku)
    promo = pd.read_csv(promo)
    sku_dict = pd.read_csv(sku_dict)

    all_groups_sku = pd.concat([first_group_sku, second_group_sku, third_group_sku])
    all_groups_sku = all_groups_sku.drop_duplicates(subset=['dates', 'SKU'])

    all_groups_sku['week_num'] = pd.to_datetime(all_groups_sku.dates).dt.isocalendar().week
    all_groups_sku['year'] = pd.to_datetime(all_groups_sku.dates).dt.year

    all_groups_sku_promo = pd.merge(all_groups_sku, promo, on=['week_num', 'year', 'SKU'], how='left')
    all_groups_sku_promo.discount = all_groups_sku_promo.discount.fillna(1)

    all_groups_sku_full = pd.merge(all_groups_sku_promo, sku_dict.rename(columns={'sku_id': 'SKU'}), on='SKU',
                                   how='left')

    all_groups_sku_full = all_groups_sku_full.sort_values(by='dates')

    all_groups_sku_full['week_num_expiration'] = pd.to_datetime(
        all_groups_sku_full.expiration_date).dt.isocalendar().week
    all_groups_sku_full['year_expiration'] = pd.to_datetime(all_groups_sku_full.expiration_date).dt.year

    all_groups_sku_full['week_num_creation'] = pd.to_datetime(all_groups_sku_full.creation_date).dt.isocalendar().week
    all_groups_sku_full['year_creation'] = pd.to_datetime(all_groups_sku_full.creation_date).dt.year

    data_full = all_groups_sku_full.sort_values(by=['dates'])
    data_full = data_full.drop_duplicates(subset=['dates', 'SKU'])

    data_full.to_csv(output, index=False)

    click.echo(f"Merged data saved to {output}")


if __name__ == '__main__':
    merge_data()
