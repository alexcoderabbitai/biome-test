import { DestroyManyResolverFactory } from 'src/engine/api/graphql/workspace-resolver-builder/factories/destroy-many-resolver.factory';
import { DestroyOneResolverFactory } from 'src/engine/api/graphql/workspace-resolver-builder/factories/destroy-one-resolver.factory';
import { RestoreManyResolverFactory } from 'src/engine/api/graphql/workspace-resolver-builder/factories/restore-many-resolver.factory';
import { SearchResolverFactory } from 'src/engine/api/graphql/workspace-resolver-builder/factories/search-resolver-factory';
import { UpdateManyResolverFactory } from 'src/engine/api/graphql/workspace-resolver-builder/factories/update-many-resolver.factory';

import { CreateManyResolverFactory } from './create-many-resolver.factory';
import { CreateOneResolverFactory } from './create-one-resolver.factory';
import { DeleteManyResolverFactory } from './delete-many-resolver.factory';
import { DeleteOneResolverFactory } from './delete-one-resolver.factory';
import { FindDuplicatesResolverFactory } from './find-duplicates-resolver.factory';
import { FindManyResolverFactory } from './find-many-resolver.factory';
import { FindOneResolverFactory } from './find-one-resolver.factory';
import { UpdateOneResolverFactory } from './update-one-resolver.factory';

export const workspaceResolverBuilderFactories = [
  FindManyResolverFactory,
  FindOneResolverFactory,
  FindDuplicatesResolverFactory,
  CreateManyResolverFactory,
  CreateOneResolverFactory,
  UpdateOneResolverFactory,
  DeleteOneResolverFactory,
  UpdateManyResolverFactory,
  DeleteManyResolverFactory,
  DestroyOneResolverFactory,
  DestroyManyResolverFactory,
  RestoreManyResolverFactory,
  SearchResolverFactory,
];

export const workspaceResolverBuilderMethodNames = {
  queries: [
    FindManyResolverFactory.methodName,
    FindOneResolverFactory.methodName,
    FindDuplicatesResolverFactory.methodName,
    SearchResolverFactory.methodName,
  ],
  mutations: [
    CreateManyResolverFactory.methodName,
    CreateOneResolverFactory.methodName,
    UpdateOneResolverFactory.methodName,
    DeleteOneResolverFactory.methodName,
    UpdateManyResolverFactory.methodName,
    DeleteManyResolverFactory.methodName,
    DestroyOneResolverFactory.methodName,
    DestroyManyResolverFactory.methodName,
    RestoreManyResolverFactory.methodName,
  ],
} as const;