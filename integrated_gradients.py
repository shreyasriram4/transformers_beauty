import torch

def integrated_gradients(model, input_ids, attention_mask, baseline, target_index, steps=50, device="cpu"):
    """
    Compute Integrated Gradients for a target prediction.
    
    Args:
        model: The trained model to evaluate.
        input_ids: Tensor of token IDs for the input text.
        attention_mask: Tensor mask indicating valid tokens.
        baseline: Tensor of baseline token IDs (e.g., all PAD tokens).
        target_index: Index of the target output for attribution.
        steps: Number of interpolation steps between baseline and input.
        device: Device to run computations on ('cpu' or 'cuda').

    Returns:
        Integrated gradients for each input embedding.
    """
    model.eval()

    # Access the embedding layer of the BERT model
    embedding_layer = model.bert.embeddings.word_embeddings

    # Convert input IDs and baseline IDs to embeddings
    input_embeddings = embedding_layer(input_ids).detach()
    baseline_embeddings = embedding_layer(baseline).detach()

    # Generate interpolation steps
    alphas = torch.linspace(0, 1, steps).to(device)
    interpolated_embeddings = [
        baseline_embeddings + alpha * (input_embeddings - baseline_embeddings)
        for alpha in alphas
    ]

    # Initialize gradients accumulator
    gradients = torch.zeros_like(input_embeddings).to(device)

    # Compute gradients for each interpolation
    for interpolated_emb in interpolated_embeddings:
        interpolated_emb.requires_grad_()
        outputs = model(
            input_ids=None, 
            attention_mask=attention_mask.unsqueeze(0), 
            inputs_embeds=interpolated_emb.unsqueeze(0)
        )
        output = outputs[0, target_index]  # Target attribute index

        # Backpropagate to compute gradients
        model.zero_grad()
        output.backward(retain_graph=True)
        gradients += interpolated_emb.grad.detach()

    # Integrated gradients: Scale by input difference
    integrated_gradients = (input_embeddings - baseline_embeddings) * gradients / steps
    return integrated_gradients.squeeze()

def explain_ingredients(model, tokenizer, text, target_index, device, steps=50):
    """
    Explain the importance of input tokens for a specific target using Integrated Gradients.
    
    Args:
        model: The trained model to evaluate.
        tokenizer: Tokenizer used to process the input text.
        text: The input text for analysis.
        target_index: Index of the target output for attribution.
        device: Device to run computations on ('cpu' or 'cuda').
        steps: Number of interpolation steps for Integrated Gradients.

    Returns:
        tokens: List of tokens before the [SEP] token.
        attributions: Array of attribution scores for each token.
    """
    model.to(device)
    model.eval()

    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128).to(device)
    input_ids = inputs['input_ids'].squeeze(0)
    attention_mask = inputs['attention_mask'].squeeze(0)

    # Create baseline with PAD tokens
    baseline_ids = torch.zeros_like(input_ids).to(device)
    baseline_ids[:] = tokenizer.pad_token_id

    # Compute Integrated Gradients
    attributions = integrated_gradients(
        model, input_ids, attention_mask, baseline_ids, target_index, steps, device
    )

    # Map attributions to tokens
    tokens = tokenizer.convert_ids_to_tokens(input_ids.cpu().numpy())

    # Focus on tokens before the [SEP]
    sep_index = tokens.index("[SEP]")
    ingredients_tokens = tokens[:sep_index]
    ingredients_attributions = attributions[:sep_index]

    # Combine and sort by importance (optional step)
    token_attributions = [(token, attr.sum().item()) for token, attr in zip(ingredients_tokens, ingredients_attributions)]
    token_attributions = sorted(token_attributions, key=lambda x: abs(x[1]), reverse=True)

    return ingredients_tokens, ingredients_attributions.cpu().numpy()
