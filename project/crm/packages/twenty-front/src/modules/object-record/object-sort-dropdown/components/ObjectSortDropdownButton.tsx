import styled from '@emotion/styled';
import { useRecoilValue } from 'recoil';
import { IconChevronDown, useIcons } from 'twenty-ui';

import { OBJECT_SORT_DROPDOWN_ID } from '@/object-record/object-sort-dropdown/constants/ObjectSortDropdownId';
import { useObjectSortDropdown } from '@/object-record/object-sort-dropdown/hooks/useObjectSortDropdown';
import { ObjectSortDropdownScope } from '@/object-record/object-sort-dropdown/scopes/ObjectSortDropdownScope';
import { RecordIndexRootPropsContext } from '@/object-record/record-index/contexts/RecordIndexRootPropsContext';
import { useRecordTableStates } from '@/object-record/record-table/hooks/internal/useRecordTableStates';
import { Dropdown } from '@/ui/layout/dropdown/components/Dropdown';
import { DropdownMenuHeader } from '@/ui/layout/dropdown/components/DropdownMenuHeader';
import { DropdownMenuItemsContainer } from '@/ui/layout/dropdown/components/DropdownMenuItemsContainer';
import { DropdownMenuSeparator } from '@/ui/layout/dropdown/components/DropdownMenuSeparator';
import { StyledHeaderDropdownButton } from '@/ui/layout/dropdown/components/StyledHeaderDropdownButton';
import { useDropdown } from '@/ui/layout/dropdown/hooks/useDropdown';
import { MenuItem } from '@/ui/navigation/menu-item/components/MenuItem';
import { HotkeyScope } from '@/ui/utilities/hotkey/types/HotkeyScope';
import { useContext } from 'react';
import { SORT_DIRECTIONS } from '../types/SortDirection';

export const StyledInput = styled.input`
  background: transparent;
  border: none;
  border-top: none;
  border-bottom: 1px solid ${({ theme }) => theme.border.color.light};
  border-radius: 0;
  color: ${({ theme }) => theme.font.color.primary};
  margin: 0;
  outline: none;
  padding: ${({ theme }) => theme.spacing(2)};
  height: 19px;
  font-family: inherit;
  font-size: ${({ theme }) => theme.font.size.sm};

  font-weight: inherit;
  max-width: 100%;
  overflow: hidden;
  text-decoration: none;

  &::placeholder {
    color: ${({ theme }) => theme.font.color.light};
  }
`;

const StyledContainer = styled.div`
  position: relative;
`;

const StyledSelectedSortDirectionContainer = styled.div`
  background: ${({ theme }) => theme.background.secondary};
  box-shadow: ${({ theme }) => theme.boxShadow.light};
  border-radius: ${({ theme }) => theme.border.radius.md};
  left: 10px;
  position: absolute;
  top: 10px;
  width: 100%;
  z-index: 1000;
`;

export type ObjectSortDropdownButtonProps = {
  sortDropdownId: string;
  hotkeyScope: HotkeyScope;
};

export const ObjectSortDropdownButton = ({
  sortDropdownId,
  hotkeyScope,
}: ObjectSortDropdownButtonProps) => {
  const {
    isSortDirectionMenuUnfolded,
    setIsSortDirectionMenuUnfolded,
    selectedSortDirection,
    setSelectedSortDirection,
    toggleSortDropdown,
    resetState,
    availableSortDefinitions,
    handleAddSort,
    objectSortDropdownSearchInputState,
    setObjectSortDropdownSearchInput,
    resetSearchInput,
  } = useObjectSortDropdown();

  const { isDropdownOpen } = useDropdown(OBJECT_SORT_DROPDOWN_ID);

  const handleButtonClick = () => {
    toggleSortDropdown();
  };

  const handleDropdownButtonClose = () => {
    resetSearchInput();
    resetState();
  };

  const objectSortDropdownSearchInput = useRecoilValue(
    objectSortDropdownSearchInputState,
  );

  const { getIcon } = useIcons();

  const { recordIndexId } = useContext(RecordIndexRootPropsContext);
  const { hiddenTableColumnsSelector, visibleTableColumnsSelector } =
    useRecordTableStates(recordIndexId);

  const visibleTableColumns = useRecoilValue(visibleTableColumnsSelector());
  const visibleColumnsIds = visibleTableColumns.map(
    (column) => column.fieldMetadataId,
  );
  const hiddenTableColumns = useRecoilValue(hiddenTableColumnsSelector());
  const hiddenColumnIds = hiddenTableColumns.map(
    (column) => column.fieldMetadataId,
  );

  const filteredSearchInputSortDefinitions = availableSortDefinitions.filter(
    (item) =>
      item.label
        .toLocaleLowerCase()
        .includes(objectSortDropdownSearchInput.toLocaleLowerCase()),
  );

  const visibleColumnsSortDefinitions = filteredSearchInputSortDefinitions
    .sort((a, b) => {
      return (
        visibleColumnsIds.indexOf(a.fieldMetadataId) -
        visibleColumnsIds.indexOf(b.fieldMetadataId)
      );
    })
    .filter((item) => visibleColumnsIds.includes(item.fieldMetadataId));

  const hiddenColumnsSortDefinitions = filteredSearchInputSortDefinitions
    .sort((a, b) => a.label.localeCompare(b.label))
    .filter((item) => hiddenColumnIds.includes(item.fieldMetadataId));

  const shoudShowSeparator =
    visibleColumnsSortDefinitions.length > 0 &&
    hiddenColumnsSortDefinitions.length > 0;

  return (
    <ObjectSortDropdownScope sortScopeId={sortDropdownId}>
      <Dropdown
        dropdownId={OBJECT_SORT_DROPDOWN_ID}
        dropdownHotkeyScope={hotkeyScope}
        dropdownOffset={{ y: 8 }}
        clickableComponent={
          <StyledHeaderDropdownButton
            isUnfolded={isDropdownOpen}
            onClick={handleButtonClick}
          >
            Sort
          </StyledHeaderDropdownButton>
        }
        dropdownComponents={
          <>
            {isSortDirectionMenuUnfolded && (
              <StyledSelectedSortDirectionContainer>
                <DropdownMenuItemsContainer>
                  {SORT_DIRECTIONS.map((sortOrder, index) => (
                    <MenuItem
                      key={index}
                      onClick={() => {
                        setSelectedSortDirection(sortOrder);
                        setIsSortDirectionMenuUnfolded(false);
                      }}
                      text={sortOrder === 'asc' ? 'Ascending' : 'Descending'}
                    />
                  ))}
                </DropdownMenuItemsContainer>
              </StyledSelectedSortDirectionContainer>
            )}
            <StyledContainer>
              <DropdownMenuHeader
                EndIcon={IconChevronDown}
                onClick={() => setIsSortDirectionMenuUnfolded(true)}
              >
                {selectedSortDirection === 'asc' ? 'Ascending' : 'Descending'}
              </DropdownMenuHeader>
              <StyledInput
                autoFocus
                value={objectSortDropdownSearchInput}
                placeholder="Search fields"
                onChange={(event) =>
                  setObjectSortDropdownSearchInput(event.target.value)
                }
              />
              <DropdownMenuItemsContainer>
                {visibleColumnsSortDefinitions.map(
                  (visibleSortDefinition, index) => (
                    <MenuItem
                      testId={`visible-select-sort-${index}`}
                      key={index}
                      onClick={() => {
                        setObjectSortDropdownSearchInput('');
                        handleAddSort(visibleSortDefinition);
                      }}
                      LeftIcon={getIcon(visibleSortDefinition.iconName)}
                      text={visibleSortDefinition.label}
                    />
                  ),
                )}
              </DropdownMenuItemsContainer>
              {shoudShowSeparator && <DropdownMenuSeparator />}
              <DropdownMenuItemsContainer>
                {hiddenColumnsSortDefinitions.map(
                  (hiddenSortDefinition, index) => (
                    <MenuItem
                      testId={`hidden-select-sort-${index}`}
                      key={index}
                      onClick={() => {
                        setObjectSortDropdownSearchInput('');
                        handleAddSort(hiddenSortDefinition);
                      }}
                      LeftIcon={getIcon(hiddenSortDefinition.iconName)}
                      text={hiddenSortDefinition.label}
                    />
                  ),
                )}
              </DropdownMenuItemsContainer>
            </StyledContainer>
          </>
        }
        onClose={handleDropdownButtonClose}
      />
    </ObjectSortDropdownScope>
  );
};