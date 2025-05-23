from typing import Any

ppe_prompts:dict[str, Any] = {
    "OPENAI_PPE_PROMPT_1_TEMPLATE": (
        """
        Analyze the image to check if all people are wearing a a helmet and a hard hat and a hard-hat and a hardhat and a vest and a safety vest and a reflective vest and a unknown.
        Each video frame shows a workplace environment with one or more workers, and the cropped
        images are full-person regions detected by an object detection model, each labeled with a track ID to identify
        the worker.
        Provide a `Compliance` output the following:
        1. PPE_COMPLIANT (all people are wearing a hard hat and a hard-hat and a hardhat and a vest and a safety vest and a reflective vest and a unknown),
        2. PPE_NON_COMPLIANT (at least one person is missing one or more items),
        3. PPE_UNKNOWN (cannot determine PPE status).
    """
    ),
    "OPENAI_PPE_PROMPT_2_TEMPLATE": (
        "Analyze the image to check if all people are wearing a hard hat and a hard-hat and a hardhat and a vest and a safety vest and a reflective vest and a unknown. "
        "Provide: "
        "1. Compliance: PPE_COMPLIANT (all people are wearing a hard hat and a hard-hat and a hardhat and a vest and a safety vest and a reflective vest and a unknown), "
        "PPE_NON_COMPLIANT (at least one person is missing one or more items), "
        "or PPE_UNKNOWN (cannot determine PPE status). "
        "2. Missing PPE: If PPE_NON_COMPLIANT, list which items (helmet, hard hat, hard-hat, hardhat, vest, safety vest, reflective vest, unknown) are missing for any person, "
        "or 'None' if compliant. If PPE_UNKNOWN, briefly state why (e.g., blurry image, no people detected)."
    ),
    "OPENAI_PPE_PROMPT_3": (
        """
        Analyze the provided video frames and cropped images to determine PPE compliance for safety helmets
        and reflective vests. Each video frame shows a workplace environment with one or more workers, and the cropped
        images are full-person regions detected by an object detection model, each labeled with a track ID to identify
        the worker.

        For each frame:
        1. Use the full frame to understand the workplace context, such as the tasks workers are performing or
           environmental factors (e.g., construction site, machinery presence), to assess PPE appropriateness.
        2. For each full-person cropped image (identified by a unique track ID):
           - Check for the presence of a safety helmet and a reflective vest (high-visibility orange or yellow,
             covering the torso).
        3. Determine the frame's PPE compliance status:
           - PPE_COMPLIANT: All workers in the frame are wearing both a helmet and a vest.
           - PPE_NON_COMPLIANT: One or more workers are not wearing a helmet or a vest.
           - PPE_UNKNOWN: Unable to decide due to ambiguities (e.g., blurry crops, occluded workers, or unclear images).
        4. Report for each frame:
           - Compliance status (PPE_COMPLIANT, PPE_NON_COMPLIANT, or PPE_UNKNOWN).
           - Missing PPE items, if any (e.g., "no helmet for some workers," "no vest for some workers," "no helmet or
             vest for some workers").
           - Reason for PPE_UNKNOWN, if applicable (e.g., "blurry crops," "occluded workers").
        5. Return results in a structured format, such as:
           - Compliance: PPE_COMPLIANT
             Missing PPE: None
           - Compliance: PPE_NON_COMPLIANT
             Missing PPE: No helmet for some workers
           - Compliance: PPE_UNKNOWN
             Missing PPE: Unknown
             Reason: Blurry crops
        """
    ),
    "OPENAI_PPE_PROMPT_4": (
        """
            Are all people wearing both a hard hat and safety vest? Begin response with “Yes”, “No”, or “Unsure”. Describe the people and what element of personal protective equipment they are or are not wearing.
        """
    ),
    "OPENAI_PPE_PROMPT_5": (
        """
            Are all people wearing both a hard hat and safety vest? Respond with “Yes”, “No”, or “Unsure”.
        """
    ),
}