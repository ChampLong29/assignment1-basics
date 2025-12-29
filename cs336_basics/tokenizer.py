from collections.abc import Iterable, Iterator
import regex as re


class Tokenizer:
    def __init__(
        self, 
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens else []
        self.bytes_to_token = {v: k for k, v in vocab.items()}

        self.merge_to_id ={}
        for i, (a, b) in enumerate(merges):
            token_id1 = self.bytes_to_token[a]
            token_id2 = self.bytes_to_token[b]
            self.merge_to_id[(token_id1, token_id2)] = (self.bytes_to_token[a + b], i)
        
        if self.special_tokens:
            sorted_special_tokens = sorted(self.special_tokens, key=lambda x: len(x), reverse=True)
            escaped_special_tokens = [re.escape(token) for token in sorted_special_tokens]
            self.special_token_pattern = re.compile(f"({'|'.join(escaped_special_tokens)})")
        else:
            self.special_token_pattern = None

        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.pat = re.compile(PAT)
        
    @classmethod
    def from_files(
        cls, 
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None
    ) -> "Tokenizer":
        import json

        with open(vocab_filepath, 'r') as f:
            vocab_data = json.load(f)

        vocab = {int(token_id_str): bytes.fromhex(hex_string)
                for token_id_str, hex_string in vocab_data.items()}
        if special_tokens:
            for special_token in special_tokens:
                if special_token.encode('utf-8') not in vocab.values():
                    vocab[len(vocab)] = special_token.encode('utf-8')
        merges = []
        with open(merges_filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    hexes = line.split(' ')
                    if len(hexes) == 2:
                        hex1, hex2 = hexes
                        merges.append((bytes.fromhex(hex1), bytes.fromhex(hex2)))
        return cls(vocab, merges, special_tokens)


    def encode(self, text: str) -> list[int]:
        text_parts = self.special_token_pattern.split(text) if self.special_token_pattern else [text]
        token_ids = []
        for part in text_parts:
            if part in self.special_tokens:
                token_ids.append(self.bytes_to_token[part.encode('utf-8')])
            elif part:
                for match in self.pat.finditer(part):
                    token_ids.extend(self._encode_word(match.group().encode('utf-8')))
        return token_ids
        
    def _encode_word(self, word_bytes: bytes) -> list[int]:
        # 将每个字节转换为对应的 token ID
        tokens = [self.bytes_to_token[bytes([b])] for b in word_bytes]
        while True:
            # 找到优先级最高的可合并 pair
            best_pair = None
            best_priority = float('inf')
            
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                if pair in self.merge_to_id:
                    new_token_id, priority = self.merge_to_id[pair]
                    if priority < best_priority:
                        best_priority = priority
                        best_pair = pair
            
            if best_pair is None:
                break
            
            # 合并所有该 pair 的非重叠出现
            new_token_id, _ = self.merge_to_id[best_pair]
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair:
                    # 找到匹配，合并
                    new_tokens.append(new_token_id)
                    i += 2  # 跳过两个 token（非重叠）
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return tokens

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        byte_list = [self.vocab[id] for id in ids]
        byte_string = b''.join(byte_list)
        return byte_string.decode('utf-8', errors='ignore')