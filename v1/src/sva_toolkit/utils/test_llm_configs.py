"""
Test script to verify LLM configurations are working correctly.
"""

import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from sva_toolkit.utils.llm_client import LLMClient, LLMConfig
from sva_toolkit.utils.file_handlers import load_json
import openai


def get_available_models(base_url: str, api_key: str) -> List[str]:
    """
    Get list of available models from an API endpoint.

    Args:
        base_url: API base URL
        api_key: API key for authentication

    Returns:
        List of available model names
    """
    try:
        client = openai.OpenAI(base_url=base_url, api_key=api_key)
        models = client.models.list()
        return [model.id for model in models.data]
    except Exception as e:
        return []


def test_llm_config(config: Dict[str, Any], test_prompt: str = "Hello, please respond with 'OK'.") -> Dict[str, Any]:
    """
    Test a single LLM configuration.

    Args:
        config: LLM configuration dictionary with base_url, model, and api_key
        test_prompt: Test prompt to send to the LLM

    Returns:
        Dictionary with test results including status, response, error message, available models, and response delay
    """
    result: Dict[str, Any] = {
        "config": config,
        "status": "unknown",
        "response": None,
        "error": None,
        "available_models": [],
        "response_delay": None,
    }
    try:
        llm_config = LLMConfig(
            base_url=config["base_url"],
            model=config["model"],
            api_key=config["api_key"],
            temperature=0.1,
            max_tokens=100,
        )
        client = LLMClient(llm_config)
        start_time = time.time()
        response = client.generate(test_prompt)
        end_time = time.time()
        result["status"] = "success"
        result["response"] = response
        result["response_delay"] = end_time - start_time
        available_models = get_available_models(config["base_url"], config["api_key"])
        result["available_models"] = available_models
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        try:
            available_models = get_available_models(config["base_url"], config["api_key"])
            result["available_models"] = available_models
        except Exception:
            pass
    return result


def test_all_llm_configs(config_file: str, test_prompt: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Test all LLM configurations from a JSON file.

    Args:
        config_file: Path to JSON file containing LLM configurations
        test_prompt: Optional custom test prompt (default: simple test prompt)

    Returns:
        List of test results for each configuration
    """
    if test_prompt is None:
        test_prompt = "Hello, please respond with 'OK'."
    configs = load_json(config_file)
    if not isinstance(configs, list):
        raise ValueError(f"Expected a list of configurations, got {type(configs)}")
    results: List[Dict[str, Any]] = []
    for i, config in enumerate(configs):
        print(f"Testing LLM {i+1}/{len(configs)}: {config.get('model', 'unknown')} at {config.get('base_url', 'unknown')}...")
        result = test_llm_config(config, test_prompt)
        results.append(result)
        if result["status"] == "success":
            delay = result.get("response_delay")
            delay_str = f"{delay:.2f}s" if delay is not None else "N/A"
            print(f"  ✓ Success: {result['response'][:100]}...")
            print(f"  ✓ Response delay: {delay_str}")
            if result.get("available_models"):
                print(f"  ✓ Available models: {len(result['available_models'])} models found")
        else:
            print(f"  ✗ Error: {result['error']}")
            if result.get("available_models"):
                print(f"  ✓ Available models: {len(result['available_models'])} models found (API accessible)")
    return results


def print_summary(results: List[Dict[str, Any]]) -> None:
    """
    Print a summary of test results.

    Args:
        results: List of test results
    """
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    success_count = sum(1 for r in results if r["status"] == "success")
    error_count = len(results) - success_count
    print(f"Total configurations tested: {len(results)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {error_count}")
    print("\nDetailed Results:")
    for i, result in enumerate(results):
        config = result["config"]
        print(f"\n{i+1}. Model: {config.get('model', 'unknown')}")
        print(f"   Base URL: {config.get('base_url', 'unknown')}")
        print(f"   Status: {result['status']}")
        if result["status"] == "success":
            response_preview = result["response"][:150] if result["response"] else "No response"
            delay = result.get("response_delay")
            delay_str = f"{delay:.2f}s" if delay is not None else "N/A"
            print(f"   Response preview: {response_preview}...")
            print(f"   Response delay: {delay_str}")
        else:
            print(f"   Error: {result['error']}")
        if result.get("available_models"):
            models = result["available_models"]
            print(f"   Available models ({len(models)}):")
            for model in models:
                marker = " ← configured" if model == config.get("model") else ""
                print(f"     - {model}{marker}")
        else:
            print(f"   Available models: Could not retrieve model list")
    print("="*80)


def main() -> int:
    """
    Main entry point for the test script.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    if len(sys.argv) < 2:
        config_file = str(Path(__file__).parent.parent.parent.parent / "examples" / "llm_configs.json")
        print(f"Using default config file: {config_file}")
    else:
        config_file = sys.argv[1]
    if not Path(config_file).exists():
        print(f"Error: Config file not found: {config_file}")
        return 1
    try:
        results = test_all_llm_configs(config_file)
        print_summary(results)
        all_success = all(r["status"] == "success" for r in results)
        return 0 if all_success else 1
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

