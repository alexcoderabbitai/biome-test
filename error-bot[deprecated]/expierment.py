from unidiff import PatchSet

def apply_patch_to_content(diff: str, original_content: str) -> str:
    # Create a PatchSet object from the diff
    patch_set = PatchSet(diff)

    # Split the original content into lines
    original_lines = original_content.splitlines(keepends=True)

    # Iterate over each file in the patch set (even though there's just one file in this case)
    for patched_file in patch_set:
        for hunk in patched_file:
            # Adjusting the start line
            original_start = hunk.source_start - 1
            hunk_lines = []

            # Iterate over the lines in the hunk and apply the changes
            for line in hunk:
                if line.is_added:
                    hunk_lines.append(line.value)
                elif not line.is_removed:
                    hunk_lines.append(original_lines.pop(0))
                elif line.is_removed:
                    original_lines.pop(0)  # Skip the removed line from the original file

            # Insert the hunk lines back into the original content
            original_lines[original_start:original_start] = hunk_lines

    # Join the modified content back into a string
    return ''.join(original_lines)

# Example usage:
diff = '''--- original
+++ modified
@@ -53,8 +53,7 @@
     async def list_memory_banks(self) -> List[MemoryBankDefWithProvider]:
         async with httpx.AsyncClient() as client:
             response = await client.get(
-                f"{self.base_url}/memory_banks/list",
-                headers={"Content-Type": "application/json"},
+                f"{self.base_url}/memory_banks/list"
             )
             response.raise_for_status()
             return [deserialize_memory_bank_def(x) for x in response.json()]
@@ -67,9 +66,7 @@
                 f"{self.base_url}/memory_banks/register",
                 json={
                     "memory_bank": json.loads(memory_bank.json()),
-                },
-                headers={"Content-Type": "application/json"},
-            )
+                })
             response.raise_for_status()

     async def get_memory_bank(
@@ -80,9 +77,7 @@
                 f"{self.base_url}/memory_banks/get",
                 params={
                     "identifier": identifier,
-                },
-                headers={"Content-Type": "application/json"},
-            )
+                })
             response.raise_for_status()
             j = response.json()
             return deserialize_memory_bank_def(j)
@@ -91,7 +86,7 @@
 async def run_main(host: str, port: int, stream: bool):
     client = MemoryBanksClient(f"http://{host}:{port}")

-    response = await client.list_memory_banks()
+    response = await client.list_memory_banks()  # This may raise an uncaught exception
     cprint(f"list_memory_banks response={response}", "green")

     # register memory bank for the first time
@@ -101,12 +96,12 @@
             embedding_model="all-MiniLM-L6-v2",
             chunk_size_in_tokens=512,
             overlap_size_in_tokens=64,
-        )
+        )  # This may raise an uncaught exception
     )
     cprint(f"register_memory_bank response={response}", "blue")

     # list again after registering
-    response = await client.list_memory_banks()
+    response = await client.list_memory_banks()  # This may raise an uncaught exception
     cprint(f"list_memory_banks response={response}", "green")'''

original_content = '''# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import json

from typing import Any, Dict, List, Optional

import fire
import httpx
from termcolor import cprint

from .memory_banks import *  # noqa: F403

def deserialize_memory_bank_def(
    j: Optional[Dict[str, Any]]
) -> MemoryBankDefWithProvider:
    if j is None:
        return None

    if "type" not in j:
        raise ValueError("Memory bank type not specified")
    type = j["type"]
    if type == MemoryBankType.vector.value:
        return VectorMemoryBankDef(**j)
    elif type == MemoryBankType.keyvalue.value:
        return KeyValueMemoryBankDef(**j)
    elif type == MemoryBankType.keyword.value:
        return KeywordMemoryBankDef(**j)
    elif type == MemoryBankType.graph.value:
        return GraphMemoryBankDef(**j)
    else:
        raise ValueError(f"Unknown memory bank type: {type}")

class MemoryBanksClient(MemoryBanks):
    def __init__(self, base_url: str):
        self.base_url = base_url

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    async def list_memory_banks(self) -> List[MemoryBankDefWithProvider]:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/memory_banks/list",
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            return [deserialize_memory_bank_def(x) for x in response.json()]

    async def register_memory_bank(
        self, memory_bank: MemoryBankDefWithProvider
    ) -> None:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/memory_banks/register",
                json={
                    "memory_bank": json.loads(memory_bank.json()),
                },
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()

    async def get_memory_bank(
        self,
        identifier: str,
    ) -> Optional[MemoryBankDefWithProvider]:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/memory_banks/get",
                params={
                    "identifier": identifier,
                },
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            j = response.json()
            return deserialize_memory_bank_def(j)

async def run_main(host: str, port: int, stream: bool):
    client = MemoryBanksClient(f"http://{host}:{port}")

    response = await client.list_memory_banks()
    cprint(f"list_memory_banks response={response}", "green")

    # register memory bank for the first time
    response = await client.register_memory_bank(
        VectorMemoryBankDef(
            identifier="test_bank2",
            embedding_model="all-MiniLM-L6-v2",
            chunk_size_in_tokens=512,
            overlap_size_in_tokens=64,
        )
    )
    cprint(f"register_memory_bank response={response}", "blue")

    # list again after registering
    response = await client.list_memory_banks()
    cprint(f"list_memory_banks response={response}", "green")

def main(host: str, port: int, stream: bool = True):
    asyncio.run(run_main(host, port, stream))

if __name__ == "__main__":
    fire.Fire(main)
'''

# Apply the patch

if __name__ == "__main__":
    modified_content = apply_patch_to_content(diff, original_content)
    print(modified_content)
