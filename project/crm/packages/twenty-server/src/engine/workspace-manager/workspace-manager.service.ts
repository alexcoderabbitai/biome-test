import { Injectable } from '@nestjs/common';

import { FeatureFlagKey } from 'src/engine/core-modules/feature-flag/enums/feature-flag-key.enum';
import { FeatureFlagService } from 'src/engine/core-modules/feature-flag/services/feature-flag.service';
import { DataSourceEntity } from 'src/engine/metadata-modules/data-source/data-source.entity';
import { DataSourceService } from 'src/engine/metadata-modules/data-source/data-source.service';
import { ObjectMetadataService } from 'src/engine/metadata-modules/object-metadata/object-metadata.service';
import { WorkspaceMigrationService } from 'src/engine/metadata-modules/workspace-migration/workspace-migration.service';
import { WorkspaceDataSourceService } from 'src/engine/workspace-datasource/workspace-datasource.service';
import { demoObjectsPrefillData } from 'src/engine/workspace-manager/demo-objects-prefill-data/demo-objects-prefill-data';
import { standardObjectsPrefillData } from 'src/engine/workspace-manager/standard-objects-prefill-data/standard-objects-prefill-data';
import { WorkspaceSyncMetadataService } from 'src/engine/workspace-manager/workspace-sync-metadata/workspace-sync-metadata.service';

@Injectable()
export class WorkspaceManagerService {
  constructor(
    private readonly workspaceDataSourceService: WorkspaceDataSourceService,
    private readonly workspaceMigrationService: WorkspaceMigrationService,
    private readonly objectMetadataService: ObjectMetadataService,
    private readonly dataSourceService: DataSourceService,
    private readonly workspaceSyncMetadataService: WorkspaceSyncMetadataService,
    private readonly featureFlagService: FeatureFlagService,
  ) {}

  /**
   * Init a workspace by creating a new data source and running all migrations
   * @param workspaceId
   * @returns Promise<void>
   */
  public async init(workspaceId: string): Promise<void> {
    const schemaName =
      await this.workspaceDataSourceService.createWorkspaceDBSchema(
        workspaceId,
      );

    const dataSourceMetadata =
      await this.dataSourceService.createDataSourceMetadata(
        workspaceId,
        schemaName,
      );

    await this.workspaceSyncMetadataService.synchronize({
      workspaceId,
      dataSourceId: dataSourceMetadata.id,
    });

    await this.prefillWorkspaceWithStandardObjects(
      dataSourceMetadata,
      workspaceId,
    );
  }

  /**
   * InitDemo a workspace by creating a new data source and running all migrations
   * @param workspaceId
   * @returns Promise<void>
   */
  public async initDemo(workspaceId: string): Promise<void> {
    const schemaName =
      await this.workspaceDataSourceService.createWorkspaceDBSchema(
        workspaceId,
      );

    const dataSourceMetadata =
      await this.dataSourceService.createDataSourceMetadata(
        workspaceId,
        schemaName,
      );

    await this.workspaceSyncMetadataService.synchronize({
      workspaceId,
      dataSourceId: dataSourceMetadata.id,
    });

    await this.prefillWorkspaceWithDemoObjects(dataSourceMetadata, workspaceId);
  }

  /**
   *
   * We are prefilling a few standard objects with data to make it easier for the user to get started.
   *
   * @param dataSourceMetadata
   * @param workspaceId
   */
  private async prefillWorkspaceWithStandardObjects(
    dataSourceMetadata: DataSourceEntity,
    workspaceId: string,
  ) {
    const workspaceDataSource =
      await this.workspaceDataSourceService.connectToWorkspaceDataSource(
        workspaceId,
      );

    if (!workspaceDataSource) {
      throw new Error('Could not connect to workspace data source');
    }

    const createdObjectMetadata =
      await this.objectMetadataService.findManyWithinWorkspace(workspaceId);

    const isWorkflowEnabled = await this.featureFlagService.isFeatureEnabled(
      FeatureFlagKey.IsWorkflowEnabled,
      workspaceId,
    );

    await standardObjectsPrefillData(
      workspaceDataSource,
      dataSourceMetadata.schema,
      createdObjectMetadata,
      isWorkflowEnabled,
    );
  }

  /**
   *
   * We are prefilling a few demo objects with data to make it easier for the user to get started.
   *
   * @param dataSourceMetadata
   * @param workspaceId
   */
  private async prefillWorkspaceWithDemoObjects(
    dataSourceMetadata: DataSourceEntity,
    workspaceId: string,
  ) {
    const workspaceDataSource =
      await this.workspaceDataSourceService.connectToWorkspaceDataSource(
        workspaceId,
      );

    if (!workspaceDataSource) {
      throw new Error('Could not connect to workspace data source');
    }

    const createdObjectMetadata =
      await this.objectMetadataService.findManyWithinWorkspace(workspaceId);

    const isWorkflowEnabled = await this.featureFlagService.isFeatureEnabled(
      FeatureFlagKey.IsWorkflowEnabled,
      workspaceId,
    );

    await demoObjectsPrefillData(
      workspaceDataSource,
      dataSourceMetadata.schema,
      createdObjectMetadata,
      isWorkflowEnabled,
    );
  }

  /**
   *
   * Delete a workspace by deleting all metadata and the schema
   *
   * @param workspaceId
   */
  public async delete(workspaceId: string): Promise<void> {
    // Delete data from metadata tables
    await this.objectMetadataService.deleteObjectsMetadata(workspaceId);
    await this.workspaceMigrationService.deleteAllWithinWorkspace(workspaceId);
    await this.dataSourceService.delete(workspaceId);
    // Delete schema
    await this.workspaceDataSourceService.deleteWorkspaceDBSchema(workspaceId);
  }
}