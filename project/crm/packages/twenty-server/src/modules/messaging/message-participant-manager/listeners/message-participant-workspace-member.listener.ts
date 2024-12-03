import { Injectable } from '@nestjs/common';
import { OnEvent } from '@nestjs/event-emitter';
import { InjectRepository } from '@nestjs/typeorm';

import { Repository } from 'typeorm';

import {
  Workspace,
  WorkspaceActivationStatus,
} from 'src/engine/core-modules/workspace/workspace.entity';
import { ObjectRecordCreateEvent } from 'src/engine/core-modules/event-emitter/types/object-record-create.event';
import { ObjectRecordUpdateEvent } from 'src/engine/core-modules/event-emitter/types/object-record-update.event';
import { objectRecordChangedProperties as objectRecordUpdateEventChangedProperties } from 'src/engine/core-modules/event-emitter/utils/object-record-changed-properties.util';
import { InjectMessageQueue } from 'src/engine/core-modules/message-queue/decorators/message-queue.decorator';
import { MessageQueue } from 'src/engine/core-modules/message-queue/message-queue.constants';
import { MessageQueueService } from 'src/engine/core-modules/message-queue/services/message-queue.service';
import { WorkspaceEventBatch } from 'src/engine/workspace-event-emitter/workspace-event.type';
import {
  MessageParticipantMatchParticipantJob,
  MessageParticipantMatchParticipantJobData,
} from 'src/modules/messaging/message-participant-manager/jobs/message-participant-match-participant.job';
import {
  MessageParticipantUnmatchParticipantJob,
  MessageParticipantUnmatchParticipantJobData,
} from 'src/modules/messaging/message-participant-manager/jobs/message-participant-unmatch-participant.job';
import { WorkspaceMemberWorkspaceEntity } from 'src/modules/workspace-member/standard-objects/workspace-member.workspace-entity';

@Injectable()
export class MessageParticipantWorkspaceMemberListener {
  constructor(
    @InjectMessageQueue(MessageQueue.messagingQueue)
    private readonly messageQueueService: MessageQueueService,
    @InjectRepository(Workspace, 'core')
    private readonly workspaceRepository: Repository<Workspace>,
  ) {}

  @OnEvent('workspaceMember.created')
  async handleCreatedEvent(
    payload: WorkspaceEventBatch<
      ObjectRecordCreateEvent<WorkspaceMemberWorkspaceEntity>
    >,
  ) {
    const workspace = await this.workspaceRepository.findOneBy({
      id: payload.workspaceId,
    });

    if (
      !workspace ||
      workspace.activationStatus !== WorkspaceActivationStatus.ACTIVE
    ) {
      return;
    }

    for (const eventPayload of payload.events) {
      if (!eventPayload.properties.after.userEmail) {
        continue;
      }

      await this.messageQueueService.add<MessageParticipantMatchParticipantJobData>(
        MessageParticipantMatchParticipantJob.name,
        {
          workspaceId: payload.workspaceId,
          email: eventPayload.properties.after.userEmail,
          workspaceMemberId: eventPayload.recordId,
        },
      );
    }
  }

  @OnEvent('workspaceMember.updated')
  async handleUpdatedEvent(
    payload: WorkspaceEventBatch<
      ObjectRecordUpdateEvent<WorkspaceMemberWorkspaceEntity>
    >,
  ) {
    for (const eventPayload of payload.events) {
      if (
        objectRecordUpdateEventChangedProperties<WorkspaceMemberWorkspaceEntity>(
          eventPayload.properties.before,
          eventPayload.properties.after,
        ).includes('userEmail')
      ) {
        await this.messageQueueService.add<MessageParticipantUnmatchParticipantJobData>(
          MessageParticipantUnmatchParticipantJob.name,
          {
            workspaceId: payload.workspaceId,
            email: eventPayload.properties.before.userEmail,
            personId: eventPayload.recordId,
          },
        );

        await this.messageQueueService.add<MessageParticipantMatchParticipantJobData>(
          MessageParticipantMatchParticipantJob.name,
          {
            workspaceId: payload.workspaceId,
            email: eventPayload.properties.after.userEmail,
            workspaceMemberId: eventPayload.recordId,
          },
        );
      }
    }
  }
}