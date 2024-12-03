import { Injectable } from '@nestjs/common';

import {
  WorkflowRunOutput,
  WorkflowRunStatus,
} from 'src/modules/workflow/common/standard-objects/workflow-run.workspace-entity';
import { WorkflowActionFactory } from 'src/modules/workflow/workflow-executor/factories/workflow-action.factory';
import { WorkflowStep } from 'src/modules/workflow/workflow-executor/types/workflow-action.type';
import { resolveInput } from 'src/modules/workflow/workflow-executor/utils/variable-resolver.util';

const MAX_RETRIES_ON_FAILURE = 3;

export type WorkflowExecutorOutput = {
  steps: WorkflowRunOutput['steps'];
  status: WorkflowRunStatus;
};

@Injectable()
export class WorkflowExecutorWorkspaceService {
  constructor(private readonly workflowActionFactory: WorkflowActionFactory) {}

  async execute({
    currentStepIndex,
    steps,
    context,
    output,
    attemptCount = 1,
  }: {
    currentStepIndex: number;
    steps: WorkflowStep[];
    output: WorkflowExecutorOutput;
    context: Record<string, unknown>;
    attemptCount?: number;
  }): Promise<WorkflowExecutorOutput> {
    if (currentStepIndex >= steps.length) {
      return { ...output, status: WorkflowRunStatus.COMPLETED };
    }

    const step = steps[currentStepIndex];

    const workflowAction = this.workflowActionFactory.get(step.type);

    const actionPayload = resolveInput(step.settings.input, context);

    const result = await workflowAction.execute(actionPayload);

    const stepOutput = output.steps[step.id];

    const error =
      result.error?.errorMessage ??
      (result.result ? undefined : 'Execution result error, no data or error');

    const updatedStepOutput = {
      id: step.id,
      name: step.name,
      type: step.type,
      outputs: [
        ...(stepOutput?.outputs ?? []),
        {
          attemptCount,
          result: result.result,
          error,
        },
      ],
    };

    const updatedOutput = {
      ...output,
      steps: {
        ...output.steps,
        [step.id]: updatedStepOutput,
      },
    };

    if (result.result) {
      return await this.execute({
        currentStepIndex: currentStepIndex + 1,
        steps,
        context: {
          ...context,
          [step.id]: result.result,
        },
        output: updatedOutput,
      });
    }

    if (step.settings.errorHandlingOptions.continueOnFailure.value) {
      return await this.execute({
        currentStepIndex: currentStepIndex + 1,
        steps,
        context,
        output: updatedOutput,
      });
    }

    if (
      step.settings.errorHandlingOptions.retryOnFailure.value &&
      attemptCount < MAX_RETRIES_ON_FAILURE
    ) {
      return await this.execute({
        currentStepIndex,
        steps,
        context,
        output: updatedOutput,
        attemptCount: attemptCount + 1,
      });
    }

    return { ...updatedOutput, status: WorkflowRunStatus.FAILED };
  }
}