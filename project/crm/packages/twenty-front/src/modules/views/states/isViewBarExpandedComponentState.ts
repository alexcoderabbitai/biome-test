import { createComponentStateV2 } from '@/ui/utilities/state/component-state/utils/createComponentStateV2';
import { ViewComponentInstanceContext } from '@/views/states/contexts/ViewComponentInstanceContext';

export const isViewBarExpandedComponentState = createComponentStateV2<boolean>({
  key: 'isViewBarExpandedComponentState',
  defaultValue: true,
  componentInstanceContext: ViewComponentInstanceContext,
});