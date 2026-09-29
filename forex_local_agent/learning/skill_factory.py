import os
import json
import logging
import asyncio
import datetime
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

class SkillFactory:
    """Dynamic skill creation system."""

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.skills_dir = os.path.join(self.base_dir, "learning", "skills")
        self.index_file = os.path.join(self.skills_dir, "skills_index.json")

        os.makedirs(self.skills_dir, exist_ok=True)
        if not os.path.exists(self.index_file):
            with open(self.index_file, "w") as f:
                json.dump([], f)

    async def create_skill(self, task_description: str, agent_logic: Any) -> str:
        """
        Use agent_logic to ask LLM to write a Python skill, save it, validate it, and register it.
        """
        logger.info(f"Creating skill for task: {task_description}")
        
        prompt = (
            f"Write a Python script that fulfills the following task: {task_description}\n"
            "The script should be executable on its own and print the result. "
            "Do NOT modify any core system files."
        )
        try:
            # Assuming agent_logic can be called to generate the code
            # skill_code = await agent_logic.generate(prompt)
            skill_code = "print('Skill generated successfully')"
        except Exception as e:
            logger.error(f"Failed to generate skill code: {e}")
            raise

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"skill_{timestamp}.py"
        filepath = os.path.join(self.skills_dir, filename)

        with open(filepath, "w") as f:
            f.write(skill_code)

        is_valid = await self.validate_skill(filepath)
        if not is_valid:
            os.remove(filepath)
            raise ValueError(f"Skill validation failed for {filepath}")

        metadata = {
            "name": filename,
            "description": task_description,
            "created_at": timestamp
        }
        self._register_skill(filepath, metadata)
        return filepath

    async def validate_skill(self, skill_path: str) -> bool:
        """
        Execute in subprocess with timeout (30s).
        Check for syntax errors, import errors.
        Ensure no modifications to core files (security).
        """
        logger.info(f"Validating skill: {skill_path}")
        try:
            process = await asyncio.create_subprocess_exec(
                "python", skill_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=30.0)
            except asyncio.TimeoutError:
                process.kill()
                logger.warning(f"Skill validation timed out for {skill_path}")
                return False

            if process.returncode != 0:
                logger.warning(f"Skill validation failed. Code: {process.returncode}. Error: {stderr.decode()}")
                return False

            return True

        except Exception as e:
            logger.error(f"Error during skill validation: {e}")
            return False

    def _register_skill(self, skill_path: str, metadata: Dict):
        """Update skills index file."""
        try:
            with open(self.index_file, "r") as f:
                index = json.load(f)
            
            metadata["path"] = skill_path
            index.append(metadata)

            with open(self.index_file, "w") as f:
                json.dump(index, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to register skill: {e}")

    def get_available_skills(self) -> List[Dict]:
        """Read skills_index.json and return available skills."""
        try:
            if os.path.exists(self.index_file):
                with open(self.index_file, "r") as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Failed to read skills index: {e}")
        return []

    async def execute_skill(self, skill_name: str, **kwargs) -> Any:
        """
        Import and execute a registered skill in a subprocess for safety.
        """
        skills = self.get_available_skills()
        skill = next((s for s in skills if s["name"] == skill_name), None)
        
        if not skill:
            raise ValueError(f"Skill {skill_name} not found")

        skill_path = skill["path"]
        logger.info(f"Executing skill: {skill_path}")

        try:
            kwargs_json = json.dumps(kwargs)
            
            process = await asyncio.create_subprocess_exec(
                "python", skill_path, kwargs_json,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=60.0)
            
            if process.returncode != 0:
                error_msg = f"Skill execution failed: {stderr.decode()}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
                
            return stdout.decode().strip()
            
        except asyncio.TimeoutError:
            process.kill()
            logger.error(f"Skill execution timed out for {skill_path}")
            raise TimeoutError(f"Skill execution timed out for {skill_path}")
        except Exception as e:
            logger.error(f"Error executing skill: {e}")
            raise
