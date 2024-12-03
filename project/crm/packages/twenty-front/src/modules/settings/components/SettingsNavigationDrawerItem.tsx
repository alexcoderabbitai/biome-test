import { useMatch, useResolvedPath } from 'react-router-dom';

import { getSettingsPagePath } from '@/settings/utils/getSettingsPagePath';
import { SettingsPath } from '@/types/SettingsPath';
import {
  NavigationDrawerItem,
  NavigationDrawerItemProps,
} from '@/ui/navigation/navigation-drawer/components/NavigationDrawerItem';
import { NavigationDrawerSubItemState } from '@/ui/navigation/navigation-drawer/types/NavigationDrawerSubItemState';

type SettingsNavigationDrawerItemProps = Pick<
  NavigationDrawerItemProps,
  'Icon' | 'label' | 'indentationLevel' | 'soon'
> & {
  matchSubPages?: boolean;
  path: SettingsPath;
  subItemState?: NavigationDrawerSubItemState;
};

export const SettingsNavigationDrawerItem = ({
  Icon,
  label,
  indentationLevel,
  matchSubPages = true,
  path,
  soon,
  subItemState,
}: SettingsNavigationDrawerItemProps) => {
  const href = getSettingsPagePath(path);
  const pathName = useResolvedPath(href).pathname;

  const isActive = !!useMatch({
    path: pathName,
    end: !matchSubPages,
  });

  return (
    <NavigationDrawerItem
      indentationLevel={indentationLevel}
      subItemState={subItemState}
      label={label}
      to={href}
      Icon={Icon}
      active={isActive}
      soon={soon}
    />
  );
};