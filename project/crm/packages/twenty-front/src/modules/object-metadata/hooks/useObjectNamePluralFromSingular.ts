import { useRecoilValue } from 'recoil';

import { currentWorkspaceState } from '@/auth/states/currentWorkspaceState';
import { objectMetadataItemFamilySelector } from '@/object-metadata/states/objectMetadataItemFamilySelector';
import { WorkspaceActivationStatus } from '~/generated/graphql';
import { generatedMockObjectMetadataItems } from '~/testing/mock-data/generatedMockObjectMetadataItems';
import { isDefined } from '~/utils/isDefined';

export const useObjectNamePluralFromSingular = ({
  objectNameSingular,
}: {
  objectNameSingular: string;
}) => {
  const currentWorkspace = useRecoilValue(currentWorkspaceState);

  let objectMetadataItem = useRecoilValue(
    objectMetadataItemFamilySelector({
      objectName: objectNameSingular,
      objectNameType: 'singular',
    }),
  );

  if (currentWorkspace?.activationStatus !== WorkspaceActivationStatus.Active) {
    objectMetadataItem =
      generatedMockObjectMetadataItems.find(
        (objectMetadataItem) =>
          objectMetadataItem.nameSingular === objectNameSingular,
      ) ?? null;
  }

  if (!isDefined(objectMetadataItem)) {
    throw new Error(
      `Object metadata item not found for ${objectNameSingular} object`,
    );
  }

  return { objectNamePlural: objectMetadataItem.namePlural };
};