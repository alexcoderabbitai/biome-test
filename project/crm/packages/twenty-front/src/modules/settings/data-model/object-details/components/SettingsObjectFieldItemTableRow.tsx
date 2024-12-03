import { useTheme } from '@emotion/react';
import styled from '@emotion/styled';
import { useMemo } from 'react';
import { IconMinus, IconPlus, isDefined, useIcons } from 'twenty-ui';

import { useGetRelationMetadata } from '@/object-metadata/hooks/useGetRelationMetadata';
import { FieldMetadataItem } from '@/object-metadata/types/FieldMetadataItem';
import { getObjectSlug } from '@/object-metadata/utils/getObjectSlug';
import { isFieldTypeSupportedInSettings } from '@/settings/data-model/utils/isFieldTypeSupportedInSettings';
import { TableCell } from '@/ui/layout/table/components/TableCell';
import { TableRow } from '@/ui/layout/table/components/TableRow';

import { RELATION_TYPES } from '../../constants/RelationTypes';

import { LABEL_IDENTIFIER_FIELD_METADATA_TYPES } from '@/object-metadata/constants/LabelIdentifierFieldMetadataTypes';
import { useFieldMetadataItem } from '@/object-metadata/hooks/useFieldMetadataItem';
import { useUpdateOneObjectMetadataItem } from '@/object-metadata/hooks/useUpdateOneObjectMetadataItem';
import { CoreObjectNameSingular } from '@/object-metadata/types/CoreObjectNameSingular';
import { getFieldSlug } from '@/object-metadata/utils/getFieldSlug';
import { isLabelIdentifierField } from '@/object-metadata/utils/isLabelIdentifierField';
import { useDeleteRecordFromCache } from '@/object-record/cache/hooks/useDeleteRecordFromCache';
import { usePrefetchedData } from '@/prefetch/hooks/usePrefetchedData';
import { PrefetchKey } from '@/prefetch/types/PrefetchKey';
import { SettingsObjectFieldActiveActionDropdown } from '@/settings/data-model/object-details/components/SettingsObjectFieldActiveActionDropdown';
import { SettingsObjectFieldInactiveActionDropdown } from '@/settings/data-model/object-details/components/SettingsObjectFieldDisabledActionDropdown';
import { settingsObjectFieldsFamilyState } from '@/settings/data-model/object-details/states/settingsObjectFieldsFamilyState';
import { LightIconButton } from '@/ui/input/button/components/LightIconButton';
import { navigationMemorizedUrlState } from '@/ui/navigation/states/navigationMemorizedUrlState';
import { View } from '@/views/types/View';
import { useNavigate } from 'react-router-dom';
import { useRecoilState } from 'recoil';
import { RelationDefinitionType } from '~/generated-metadata/graphql';
import { SettingsObjectDetailTableItem } from '~/pages/settings/data-model/types/SettingsObjectDetailTableItem';
import { SettingsObjectFieldDataType } from './SettingsObjectFieldDataType';

type SettingsObjectFieldItemTableRowProps = {
  settingsObjectDetailTableItem: SettingsObjectDetailTableItem;
  status: 'active' | 'disabled';
  mode: 'view' | 'new-field';
};

export const StyledObjectFieldTableRow = styled(TableRow)`
  grid-template-columns: 180px 148px 148px 36px;
`;

const StyledNameTableCell = styled(TableCell)`
  color: ${({ theme }) => theme.font.color.primary};
  gap: ${({ theme }) => theme.spacing(2)};
`;

const StyledNameLabel = styled.div`
  white-space: nowrap;
  text-overflow: ellipsis;
  overflow: hidden;
`;

const StyledIconTableCell = styled(TableCell)`
  justify-content: center;
  padding-right: ${({ theme }) => theme.spacing(1)};
`;

export const SettingsObjectFieldItemTableRow = ({
  settingsObjectDetailTableItem,
  mode,
  status,
}: SettingsObjectFieldItemTableRowProps) => {
  const { fieldMetadataItem, identifierType, objectMetadataItem } =
    settingsObjectDetailTableItem;

  const isRemoteObjectField = objectMetadataItem.isRemote;

  const variant = objectMetadataItem.isCustom ? 'identifier' : 'field-type';

  const navigate = useNavigate();

  const [navigationMemorizedUrl, setNavigationMemorizedUrl] = useRecoilState(
    navigationMemorizedUrlState,
  );

  const theme = useTheme();
  const { getIcon } = useIcons();
  const Icon = getIcon(fieldMetadataItem.icon);

  const getRelationMetadata = useGetRelationMetadata();

  const { relationObjectMetadataItem, relationType } =
    useMemo(
      () => getRelationMetadata({ fieldMetadataItem }),
      [fieldMetadataItem, getRelationMetadata],
    ) ?? {};

  const fieldType = fieldMetadataItem.type;
  const isFieldTypeSupported = isFieldTypeSupportedInSettings(fieldType);

  const RelationIcon = relationType
    ? RELATION_TYPES[relationType].Icon
    : undefined;

  const isLabelIdentifier = isLabelIdentifierField({
    fieldMetadataItem,
    objectMetadataItem,
  });

  const canToggleField = !isLabelIdentifier;

  const canBeSetAsLabelIdentifier =
    objectMetadataItem.isCustom &&
    !isLabelIdentifier &&
    LABEL_IDENTIFIER_FIELD_METADATA_TYPES.includes(fieldMetadataItem.type);

  const linkToNavigate = `./${getFieldSlug(fieldMetadataItem)}`;

  const {
    activateMetadataField,
    deactivateMetadataField,
    deleteMetadataField,
  } = useFieldMetadataItem();

  const { records: allViews } = usePrefetchedData<View>(PrefetchKey.AllViews);

  const deleteViewFromCache = useDeleteRecordFromCache({
    objectNameSingular: CoreObjectNameSingular.View,
  });

  const handleDisableField = async (
    activeFieldMetadatItem: FieldMetadataItem,
  ) => {
    await deactivateMetadataField(activeFieldMetadatItem);

    const deletedViewIds = allViews
      .map((view) => {
        if (view.kanbanFieldMetadataId === activeFieldMetadatItem.id) {
          deleteViewFromCache(view);
          return view.id;
        }

        return null;
      })
      .filter(isDefined);

    const [baseUrl, queryParams] = navigationMemorizedUrl.includes('?')
      ? navigationMemorizedUrl.split('?')
      : [navigationMemorizedUrl, ''];

    const params = new URLSearchParams(queryParams);
    const currentViewId = params.get('view');

    if (isDefined(currentViewId) && deletedViewIds.includes(currentViewId)) {
      params.delete('view');
      const updatedUrl = params.toString()
        ? `${baseUrl}?${params.toString()}`
        : baseUrl;
      setNavigationMemorizedUrl(updatedUrl);
    }
  };

  const { updateOneObjectMetadataItem } = useUpdateOneObjectMetadataItem();

  const handleSetLabelIdentifierField = (
    activeFieldMetadatItem: FieldMetadataItem,
  ) =>
    updateOneObjectMetadataItem({
      idToUpdate: objectMetadataItem.id,
      updatePayload: {
        labelIdentifierFieldMetadataId: activeFieldMetadatItem.id,
      },
    });

  const [, setActiveSettingsObjectFields] = useRecoilState(
    settingsObjectFieldsFamilyState({
      objectMetadataItemId: objectMetadataItem.id,
    }),
  );

  const handleToggleField = () => {
    setActiveSettingsObjectFields((previousFields) => {
      const newFields = isDefined(previousFields)
        ? previousFields?.map((field) =>
            field.id === fieldMetadataItem.id
              ? { ...field, isActive: !field.isActive }
              : field,
          )
        : null;

      return newFields;
    });
  };

  const typeLabel =
    variant === 'field-type'
      ? isRemoteObjectField
        ? 'Remote'
        : fieldMetadataItem.isCustom
          ? 'Custom'
          : 'Standard'
      : variant === 'identifier'
        ? isDefined(identifierType)
          ? identifierType === 'label'
            ? 'Record text'
            : 'Record image'
          : ''
        : '';

  if (!isFieldTypeSupported) return null;

  return (
    <StyledObjectFieldTableRow
      to={mode === 'view' ? linkToNavigate : undefined}
    >
      <StyledNameTableCell>
        {!!Icon && (
          <Icon
            style={{ minWidth: theme.icon.size.md }}
            size={theme.icon.size.md}
            stroke={theme.icon.stroke.sm}
          />
        )}
        <StyledNameLabel title={fieldMetadataItem.label}>
          {fieldMetadataItem.label}
        </StyledNameLabel>
      </StyledNameTableCell>
      <TableCell>{typeLabel}</TableCell>
      <TableCell>
        <SettingsObjectFieldDataType
          Icon={RelationIcon}
          label={
            relationType === RelationDefinitionType.ManyToOne ||
            relationType === RelationDefinitionType.OneToOne
              ? relationObjectMetadataItem?.labelSingular
              : relationObjectMetadataItem?.labelPlural
          }
          to={
            relationObjectMetadataItem?.namePlural &&
            !relationObjectMetadataItem.isSystem
              ? `/settings/objects/${getObjectSlug(relationObjectMetadataItem)}`
              : undefined
          }
          value={fieldType}
        />
      </TableCell>
      <StyledIconTableCell>
        {status === 'active' ? (
          mode === 'view' ? (
            <SettingsObjectFieldActiveActionDropdown
              isCustomField={fieldMetadataItem.isCustom === true}
              scopeKey={fieldMetadataItem.id}
              onEdit={() => navigate(linkToNavigate)}
              onSetAsLabelIdentifier={
                canBeSetAsLabelIdentifier
                  ? () => handleSetLabelIdentifierField(fieldMetadataItem)
                  : undefined
              }
              onDeactivate={
                isLabelIdentifier
                  ? undefined
                  : () => handleDisableField(fieldMetadataItem)
              }
            />
          ) : (
            canToggleField && (
              <LightIconButton
                Icon={IconMinus}
                accent="tertiary"
                onClick={handleToggleField}
              />
            )
          )
        ) : mode === 'view' ? (
          <SettingsObjectFieldInactiveActionDropdown
            isCustomField={fieldMetadataItem.isCustom === true}
            scopeKey={fieldMetadataItem.id}
            onEdit={() => navigate(linkToNavigate)}
            onActivate={() => activateMetadataField(fieldMetadataItem)}
            onDelete={() => deleteMetadataField(fieldMetadataItem)}
          />
        ) : (
          <LightIconButton
            Icon={IconPlus}
            accent="tertiary"
            onClick={handleToggleField}
          />
        )}
      </StyledIconTableCell>
    </StyledObjectFieldTableRow>
  );
};