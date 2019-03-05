import sys
import requests
import json
from tqdm import tqdm

class API_client:
    def __init__(self, verbose):
        self.verbose = verbose

    def url(self, path):
        """Provides a URL for making REST API requests.

        Returns:
            str: URL containing API root path, API command, and API key.
        """
        api_key = 'YOUR-ETSY-API-KEY-GOES-HERE'
        limit = '100'
        return f"https://openapi.etsy.com/v2/{path}?api_key={api_key}&limit={limit}"

    def get_shop(self, shop_name):
        """Gets Shop data: title and announcement.
        
        Args:
            shop_name (str): Shop name.

        Returns:
            dict: Shop data including title and announcement.
        """
        response = requests.get(self.url(f'shops/{shop_name}'))
        if self.check_response(response):
            return response.json()

    def get_shop_listings(self, lock, shop_name, position, page=1, pbar=None):
        """Retrieves data for an Etsy shop's listings.

        In case there are more listings than the pagination limit,
        recursively retrieves subsequent pages until all listings are compiled.

        Args:
            lock (_thread.lock): Thead lock for non-thread-safe tqdm progress bar updates.
            shop_name (str): Shop name.
            position (int): Position in list of names, for positioning of progress bar.
            page (int): Page for paginated results.
            pbar (tqdm._tqdm.tqdm): Progress bar for listings download.
        Returns:
            list or tuple: List of product listings. In recursive calls, also returns next page.
        """
        response = requests.get(self.url(f'shops/{shop_name}/listings/active') + f'&page={page}')
        if self.check_response(response):
            data = response.json()
            listings = data['results']
            count = data['count']
            next_page = data['pagination']['next_page']
            limit = int(data['params']['limit'])
            total_loaded = ((limit * (page - 1)) + len(listings))
            if self.verbose:
                if page == 1:
                    pbar = tqdm(
                        total=int(count),
                        desc='{0: <21}'.format(shop_name),
                        position=position
                    )
                with lock:
                    pbar.update(len(listings))
                    pbar.refresh()
            while next_page is not None:
                next_listings, next_page = self.get_shop_listings(
                    lock, shop_name, position, page=page + 1, pbar=pbar
                )
                listings += next_listings
            if page == 1:
                if self.verbose:
                    with lock:
                        pbar.close()
                return listings
            else:
                return listings, next_page

    def check_response(self, response):
        """Checks REST API response to ensure resource and API key are OK."""
        if response.status_code == 200:
            return True
        if response.status_code == 400:
            sys.exit('You must configure an API key.')
        if response.status_code == 403:
            sys.exit('Your API key is not authorized for this request.')
        if response.status_code == 404:
            sys.exit('The resource you requested is not found.')
