"""Initialize EVE data and configure a dedicated SSO client."""

import argparse
import getpass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['sde', 'fittings', 'esi-init', 'esi-configure'])
    args = parser.parse_args()
    if args.action == 'sde':
        from .eve_static_data import import_sde
        from .eve_names import refresh_names
        print(import_sde())
        refresh_names()
    elif args.action == 'fittings':
        from .eve_fittings import import_sources
        print(import_sources())
    else:
        from . import eve_esi_bindings as store
        store.ensure_schema()
        if args.action == 'esi-configure':
            client_id = input('EVE SSO Client ID: ').strip()
            secret = getpass.getpass('EVE SSO Client Secret: ')
            callback = input('HTTPS callback URL: ').strip()
            if not callback.endswith('/eve/oauth/callback'):
                raise ValueError('Callback URL must end with /eve/oauth/callback')
            store.configure_client(client_id, secret, callback)
        print('Configuration complete.')


if __name__ == '__main__':
    main()
