from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from opencc import OpenCC

from .cancellation import JobCancelled, check_cancelled
from .checkpoints import TextCheckpointStore
from .providers import ProviderError, TextProvider, generate_with_retry
from .models import OutputLanguage, TranscriptSegment


ProgressFn = Callable[[float, str], None]
CancelFn = Callable[[], bool]
SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[。！？.!?])")
MAX_SUMMARY_MERGE_ROUNDS = 8


@dataclass(frozen=True)
class TextChunk:
    index: int
    text: str
    fallback_text: str


class ResumableTextProcessor:
    def __init__(self) -> None:
        self.opencc = OpenCC("t2s")
        self.checkpoints = TextCheckpointStore()

    def polish(
        self,
        segments: list[TranscriptSegment],
        fallback_text: str,
        provider: TextProvider,
        model_name: str,
        output_language: OutputLanguage,
        chunk_chars: int,
        checkpoint_dir: Path,
        progress: ProgressFn | None = None,
        cancelled: CancelFn | None = None,
    ) -> tuple[str, list[str], list[str]]:
        chunks = self.chunk_segments(segments, chunk_chars)
        if not chunks:
            chunks = [TextChunk(index=0, text=fallback_text, fallback_text=fallback_text)]
        polished: list[str] = []
        warnings: list[str] = []
        polish_prompt = self.polish_prompt(output_language)

        for index, chunk in enumerate(chunks):
            self._check_cancel(cancelled)
            if progress:
                progress((index + 0.1) / max(1, len(chunks)), f"整理文字分块 {index + 1}/{len(chunks)}")
            cache_path = checkpoint_dir / f"polish_{index:04d}.json"
            cached = self.checkpoints.load(cache_path, chunk.text, model_name, output_language)
            if cached is not None:
                polished.append(cached)
                continue
            try:
                output = generate_with_retry(
                    provider,
                    model_name,
                    polish_prompt,
                    chunk.text,
                    retries=2,
                    cancelled=cancelled,
                ).strip()
                output = self.apply_output_language(output, output_language)
                if not self.checkpoints.save(cache_path, chunk.text, model_name, output_language, output):
                    warnings.append("文字整理已完成，但本段进度未能保存；下次继续时可能需要重新整理。")
                self._check_cancel(cancelled)
                polished.append(output)
            except ProviderError as exc:
                fallback = self.apply_output_language(chunk.fallback_text, output_language)
                polished.append(fallback)
                warnings.append(f"第 {index + 1} 个文字整理分块失败，已保留本地整理稿：{exc}")

        self._check_cancel(cancelled)
        formatted_text = "\n\n".join(part.strip() for part in polished if part.strip()).strip()
        if not formatted_text:
            formatted_text = self.apply_output_language(fallback_text, output_language)
        return formatted_text, polished, warnings

    def summarize(
        self,
        polished_chunks: list[str],
        provider: TextProvider,
        model_name: str,
        output_language: OutputLanguage,
        chunk_chars: int,
        checkpoint_dir: Path,
        progress: ProgressFn | None,
        cancelled: CancelFn | None,
        *,
        inline_visual: bool = False,
    ) -> tuple[str, list[str]]:
        warnings: list[str] = []
        partials: list[str] = []
        prompt = self.partial_summary_prompt(output_language, inline_visual=inline_visual)

        for index, text in enumerate(polished_chunks):
            self._check_cancel(cancelled)
            if not text.strip():
                continue
            if progress:
                progress(
                    (index + 0.1) / max(1, len(polished_chunks)),
                    f"生成分段摘要 {index + 1}/{len(polished_chunks)}",
                )
            cache_path = checkpoint_dir / f"summary_{index:04d}.json"
            cached = self.checkpoints.load(cache_path, text, model_name, output_language)
            if cached is not None:
                partials.append(cached)
                continue
            try:
                output = generate_with_retry(
                    provider, model_name, prompt, text, retries=2, cancelled=cancelled,
                ).strip()
                output = self.apply_output_language(output, output_language)
                if not self.checkpoints.save(cache_path, text, model_name, output_language, output):
                    warnings.append("分段摘要已完成，但本段进度未能保存；下次继续时可能需要重新总结。")
                self._check_cancel(cancelled)
                partials.append(output)
            except ProviderError as exc:
                warnings.append(f"第 {index + 1} 个摘要分块失败：{exc}")

        self._check_cancel(cancelled)
        if not partials:
            return "", warnings

        level = 0
        current = partials
        while len("\n\n".join(current)) > chunk_chars and len(current) > 1:
            self._check_cancel(cancelled)
            if level >= MAX_SUMMARY_MERGE_ROUNDS:
                warnings.append("摘要归并已达到处理上限，已保留现有分段摘要。")
                return "\n\n".join(current), warnings
            groups = self.chunk_strings(current, chunk_chars)
            merged: list[str] = []
            for index, group in enumerate(groups):
                self._check_cancel(cancelled)
                input_text = "\n\n".join(group)
                cache_path = checkpoint_dir / f"summary_merge_{level}_{index:04d}.json"
                cached = self.checkpoints.load(cache_path, input_text, model_name, output_language)
                if cached is not None:
                    merged.append(cached)
                    continue
                try:
                    output = generate_with_retry(
                        provider,
                        model_name,
                        self.merge_summary_prompt(output_language, inline_visual=inline_visual),
                        input_text,
                        retries=2,
                        cancelled=cancelled,
                    ).strip()
                    output = self.apply_output_language(output, output_language)
                    if not self.checkpoints.save(cache_path, input_text, model_name, output_language, output):
                        warnings.append("摘要归并已完成，但本段进度未能保存；下次继续时可能需要重新归并。")
                    self._check_cancel(cancelled)
                    merged.append(output)
                except ProviderError as exc:
                    warnings.append(f"摘要归并分块失败：{exc}")
                    merged.append(input_text)
            if len(merged) >= len(current) and len("\n\n".join(merged)) >= len("\n\n".join(current)):
                warnings.append("摘要归并未能缩短内容，已保留现有分段摘要。")
                return "\n\n".join(current), warnings
            current = merged
            level += 1

        self._check_cancel(cancelled)
        final_input = "\n\n".join(current)
        cache_path = checkpoint_dir / "summary_final.json"
        cached = self.checkpoints.load(cache_path, final_input, model_name, output_language)
        if cached is not None:
            return cached, warnings
        try:
            summary = generate_with_retry(
                provider,
                model_name,
                self.final_summary_prompt(output_language, inline_visual=inline_visual),
                final_input,
                retries=2,
                cancelled=cancelled,
            ).strip()
            summary = self.apply_output_language(summary, output_language)
            if not self.checkpoints.save(cache_path, final_input, model_name, output_language, summary):
                warnings.append("整篇总结已完成，但进度未能保存；下次继续时可能需要重新总结。")
            self._check_cancel(cancelled)
            return summary, warnings
        except ProviderError as exc:
            warnings.append(f"整篇摘要生成失败：{exc}")
            return "\n\n".join(current), warnings

    def chunk_segments(self, segments: list[TranscriptSegment], max_chars: int) -> list[TextChunk]:
        if max_chars <= 0:
            raise ValueError("chunk_chars 必须大于 0。")
        if not segments:
            return []

        groups: list[list[TranscriptSegment]] = []
        current: list[TranscriptSegment] = []
        current_length = 0

        for index, segment in enumerate(segments):
            text = re.sub(r"\s+", " ", segment.text).strip()
            if not text:
                continue
            pause_before = 0.0
            if current:
                pause_before = max(0.0, segment.start - current[-1].end)
            should_break = bool(
                current
                and (
                    current_length + len(text) + 1 > max_chars
                    or (pause_before >= 1.4 and current_length >= max_chars * 0.45)
                )
            )
            if should_break:
                groups.append(current)
                current = []
                current_length = 0
            current.append(segment)
            current_length += len(text) + 1

            if len(text) > max_chars:
                current = [
                    TranscriptSegment(segment.start, segment.end, part)
                    for part in self._split_long_text(text, max_chars)
                ]
                groups.extend([[item] for item in current[:-1]])
                current = current[-1:]
                current_length = len(current[0].text)

        if current:
            groups.append(current)

        chunks: list[TextChunk] = []
        for index, group in enumerate(groups):
            current_text = self._join_segment_text(group)
            input_text = current_text
            if index:
                context = self._join_segment_text(self._overlap_segments(groups[index - 1], max_chars))
                if context:
                    input_text = (
                        "<context>以下仅供衔接，不要在输出中重复：</context>\n"
                        f"{context}\n"
                        "<current>只整理下面的当前内容：</current>\n"
                        f"{current_text}"
                    )
            chunks.append(TextChunk(index=index, text=input_text, fallback_text=current_text))
        return chunks

    def chunk_strings(self, values: list[str], max_chars: int) -> list[list[str]]:
        if max_chars <= 0:
            raise ValueError("分块字符数必须大于 0。")
        groups: list[list[str]] = []
        current: list[str] = []
        length = 0
        expanded: list[str] = []
        for value in values:
            if len(value) > max_chars:
                expanded.extend(self._split_long_text(value, max_chars))
            else:
                expanded.append(value)
        for value in expanded:
            if current and length + len(value) + 2 > max_chars:
                groups.append(current)
                current = []
                length = 0
            current.append(value)
            length += len(value) + 2
        if current:
            groups.append(current)
        return groups

    def apply_output_language(self, text: str, output_language: OutputLanguage) -> str:
        if output_language == OutputLanguage.SIMPLIFIED_CHINESE and text.strip():
            return self.opencc.convert(text)
        return text

    def polish_prompt(self, language: OutputLanguage) -> str:
        language_rule = self._language_rule(language)
        return (
            "你是严谨的文字稿编辑，负责把语音转写的口语整理成通顺的书面文字稿。要求："
            "1. 严格保留原意，不新增、不删减事实，不发表评论或补充背景。"
            "2. 补全标点，按语义换行分段。"
            "3. 删除“嗯、啊、那个、然后然后”这类口头禅和明显的口误重复。"
            "4. 只纠正能由上下文确定的同音错别字，不确定就保持原样。"
            "5. 保留英文术语、产品名、代码、缩写和英文原句，英文单词之间保留空格。"
            f"{language_rule}"
            "如果输入含有 context/current 标记，只整理并输出 current 部分，不要重复 context。"
            "只输出整理后的正文，不要加任何前言、说明或标题。"
        )

    def partial_summary_prompt(self, language: OutputLanguage, *, inline_visual: bool = False) -> str:
        if inline_visual:
            return (
                "请把下面这段讲述及画面证据整理成图文解读的分段摘要，按主题用小节和自然段组织。"
                "先提炼本段的主要判断，再把同一主题的讲述和画面合并，讲清判断、依据，以及成立的条件或时间范围。"
                "只保留影响理解主要判断的信息和必要限定，不逐句复述讲述或逐张描述截图；只依据资料，不新增事实。"
                + self._visual_summary_rule()
                + self._language_rule(language)
            )
        return (
            "请总结下面这段文字，提炼 3 到 5 条关键要点，覆盖其中的核心信息和结论。"
            "只依据原文，不补充原文没有的内容，不重复啰嗦。"
            "每条要点单独一行、以“- ”开头，一句话讲清一个点。"
            "保留英文术语、产品名、代码和缩写。"
            + self._language_rule(language)
        )

    def merge_summary_prompt(self, language: OutputLanguage, *, inline_visual: bool = False) -> str:
        if inline_visual:
            return (
                "请把下面这些图文分段摘要合并成更紧凑的阶段摘要，合并同一主题、删除重复，"
                "先呈现主要判断，再按主题小节和自然段说明依据、条件和时间范围，不沿用分段顺序堆叠内容。"
                "保留影响结论的事实与不确定之处，删除旁枝和重复核对细节，不新增内容。"
                + self._visual_summary_rule()
                + self._language_rule(language)
            )
        return (
            "请把下面这些分段摘要合并成更紧凑的阶段摘要：删除重复，保留全部关键事实与结论，不新增内容。"
            "保持要点形式，每条单独一行、以“- ”开头。"
            + self._language_rule(language)
        )

    def final_summary_prompt(self, language: OutputLanguage, *, inline_visual: bool = False) -> str:
        if inline_visual:
            return (
                "请基于下面这些图文摘要写成一篇连贯的图文解读。开头直接概括全文最重要的判断及其适用范围，"
                "让读者先知道作者怎么看。随后按重要性组织主题小节和自然段，同一主题集中讲完，"
                "讲清作者的判断、关键依据，以及判断成立的条件或时间范围；短期与长期、不同标的不要混在一起。"
                "需要画面帮助理解时，在相关观点后放对应画面引用，接着用简明解释说明它与判断的关系，"
                "不要重复一遍图上所有文字。"
                "不要分成文字总结和画面笔记两块，不要列成互相孤立的要点，也不要重复同一解释。"
                "准确区分讲述原话、画面可见事实和未确认推测；不编造，保留必要的英文术语、产品名、代码和缩写。"
                + self._visual_summary_rule()
                + self._language_rule(language)
            )
        return (
            "请基于下面这些分段摘要，写出整篇内容的总结。格式："
            "先用一句话概括全文主旨；然后另起一段，按重要性列出 4 到 8 条要点，"
            "每条单独一行、以“- ”开头。"
            "要求：准确、清晰、不编造、不重复，保留英文术语、产品名、代码和缩写。"
            + self._language_rule(language)
        )

    @staticmethod
    def _visual_summary_rule() -> str:
        return (
            "资料中的 [[frame:1]] 这类标记是对应截图的引用，数字仅为画面编号。"
            "凡依据画面说明的内容，必须原样保留该画面的准确标记，将标记单独一行放在相关观点与画面解释之间。"
            "只能引用本次输入里存在的标记，不得改号、编造或把另一个画面的标记移到无关内容。"
            "同一画面只在最相关的一处引用；没有相关画面依据的文字不强行配图。"
            "只选帮助理解核心判断的截图，不必使用所有画面标记，不为展示每张已提取截图而拉长正文。"
            "已有画面笔记会作为可展开的详细依据保留；新闻逐条反驳、鼠标读数、绘图控制点、合约编号和纠错过程等核对细节，"
            "不在正文重新复述；若某项细节会改变结论，正文简洁说明它对结论的影响。"
            "保留与画面相关的视频时间，数字、预测和观点的归属不清楚时明确说明。"
            "总结作者观点，不自行补充交易建议。"
        )

    def _language_rule(self, language: OutputLanguage) -> str:
        if language == OutputLanguage.ENGLISH:
            return "全部使用英文输出。"
        if language == OutputLanguage.FOLLOW_SOURCE:
            return "跟随原文的主要语言输出，不主动翻译。"
        return "使用简体中文输出，但不要翻译必要的英文原句和术语。"

    def _overlap_segments(self, segments: list[TranscriptSegment], max_chars: int) -> list[TranscriptSegment]:
        target = min(240, max(80, max_chars // 12))
        selected: list[TranscriptSegment] = []
        length = 0
        for segment in reversed(segments):
            if selected and length + len(segment.text) > target:
                break
            selected.append(segment)
            length += len(segment.text)
        return list(reversed(selected))

    def _split_long_text(self, text: str, max_chars: int) -> list[str]:
        if max_chars <= 0:
            raise ValueError("分块字符数必须大于 0。")
        sentences = [part for part in SENTENCE_BOUNDARY_RE.split(text) if part]
        result: list[str] = []
        current = ""
        for sentence in sentences:
            # Even a sentence with few punctuation marks must fit the limit.
            # Keep whitespace so splitting never changes the original content.
            for index in range(0, len(sentence), max_chars):
                part = sentence[index : index + max_chars]
                if current and len(current) + len(part) > max_chars:
                    result.append(current)
                    current = ""
                current += part
        if current:
            result.append(current)
        return result

    def _join_segment_text(self, segments: list[TranscriptSegment]) -> str:
        result = ""
        for segment in segments:
            text = re.sub(r"\s+", " ", segment.text).strip()
            if not text:
                continue
            separator = ""
            if result and result[-1:].isascii() and text[:1].isascii():
                separator = " "
            result += separator + text
        return result

    def _check_cancel(self, cancelled: CancelFn | None) -> None:
        check_cancelled(cancelled)
